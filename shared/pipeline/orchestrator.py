"""
Pipeline orchestrator - main answer_question function and CLI entry point.
"""

import time
import logging
import json
import argparse
from typing import List, Dict, Optional

from openai import OpenAI, OpenAIError

from shared.config import AppConfig
from shared.prompt_manager import detect_intent, build_prompt
from shared.pipeline.config import Config
from shared.pipeline.models import QueryMetrics, PipelineResult, ValidationError
from shared.pipeline.preprocessing import (
    validate_question,
    extract_filters,
    normalize_query,
    rewrite_query_simple,
    rewrite_query_with_llm,
    enrich_query_with_context,
    detect_ambiguous_vehicle,
)
from shared.pipeline.retrieval import retrieve, rerank_hits
from shared.pipeline.generation import (
    build_context_and_sources,
    parse_answer_nudge,
    generate_llm_failure_fallback,
)
from shared.key_facts import KeyFacts

logger = logging.getLogger(__name__)


def answer_question(
    user_question: str,
    conversation_history: List[Dict[str, str]] = None,
    top_k: int = None,
    enable_llm_rewrite: bool = False,
    key_facts: KeyFacts = None
) -> PipelineResult:
    """Main RAG pipeline orchestrator."""
    if top_k is None:
        top_k = Config.TOP_K

    pipeline_start = time.time()
    metrics = QueryMetrics()

    logger.info("=" * 60)
    logger.info("RAG PIPELINE START")
    logger.info("=" * 60)
    logger.info(f"Question: {user_question}")

    try:
        # 1. Validate input
        user_question = validate_question(user_question)

        # 2. Detect intents EARLY (before any error returns)
        intents = detect_intent(user_question)
        logger.info(f"Detected intents: {intents}")

        # 2.5. Check for ambiguous vehicle mentions - return clarification if needed
        ambiguous = detect_ambiguous_vehicle(user_question, conversation_history)
        if ambiguous:
            logger.info(f"Ambiguous vehicle detected: {ambiguous['vehicle']}")
            # Return early with clarification question
            metrics.total_time = time.time() - pipeline_start
            return PipelineResult(
                answer=ambiguous["question"],
                sources=[],
                metrics=metrics,
                filters={},
                rewritten_queries=[],
                retrieved_chunks=[],
                intents=intents,
                confidence_score=1.0,  # High confidence - we know we need clarification
                search_time_ms=0.0,
                llm_time_ms=0.0,
                nudge=f"The {ambiguous['vehicle'].title()} comes in variants with different charging speeds.",
                delayed_nudge=None
            )

        # 3. Preprocess: extract filters + normalize
        preprocess_start = time.time()
        clean_query, filters = extract_filters(user_question)

        # Enrich vague queries with context from conversation history
        enriched_query = enrich_query_with_context(clean_query, conversation_history)

        normalized_query = normalize_query(enriched_query)
        deterministic_rewrite = rewrite_query_simple(normalized_query)
        metrics.preprocessing_time = time.time() - preprocess_start

        logger.info(f"Original query: {clean_query}")
        if enriched_query != clean_query:
            logger.info(f"Enriched query: {enriched_query}")
        logger.info(f"Deterministic rewrite: {deterministic_rewrite}")
        if filters:
            logger.info(f"Filters: {filters}")

        # 4. Optionally generate LLM rewrites
        rewritten_queries = [deterministic_rewrite]

        if enable_llm_rewrite:
            llm_rewrite_start = time.time()
            try:
                llm_rewrites = rewrite_query_with_llm(
                    user_question,
                    max_variants=Config.MAX_REWRITE_VARIANTS
                )

                for rq in llm_rewrites:
                    if rq not in rewritten_queries:
                        rewritten_queries.append(rq)

                logger.info(f"LLM rewrites: {llm_rewrites}")

            except Exception as e:
                logger.warning(f"LLM rewrite failed, using deterministic only: {e}")

            metrics.llm_rewrite_time = time.time() - llm_rewrite_start

        logger.info(f"Final query variants: {rewritten_queries}")

        # 5. Retrieve candidates
        retrieval_start = time.time()
        all_hits = retrieve(user_question, rewritten_queries, filters=filters)
        metrics.retrieval_time = time.time() - retrieval_start
        metrics.search_time_ms = metrics.retrieval_time * 1000
        metrics.retrieved_count = len(all_hits)

        if not all_hits:
            logger.warning("No results retrieved from vector search - continuing with LLM only")

        # 6. Rerank
        rerank_start = time.time()
        reranked_hits = rerank_hits(user_question, all_hits)
        metrics.rerank_time = time.time() - rerank_start
        metrics.reranked_count = len(reranked_hits)

        # 7. Select top-k and build context
        top_hits = reranked_hits[:top_k]
        context, sources, context_tokens = build_context_and_sources(
            top_hits,
            max_tokens=Config.MAX_CONTEXT_TOKENS
        )

        metrics.chunks_used = len(top_hits)
        metrics.context_tokens = context_tokens

        if not context:
            logger.warning("No context could be built from results - will use conversation history only")

        # 8. Generate answer with conversation history
        generation_start = time.time()

        # Build dynamic system prompt with context
        system_prompt = build_prompt(
            intents=intents,
            question=user_question,
            context=context,
            conversation_history=conversation_history,
            key_facts=key_facts
        )

        logger.info("Using dynamic system prompt with RAG context")

        # Build messages array
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history if provided
        if conversation_history:
            history_limit = AppConfig.CONVERSATION_HISTORY_LIMIT
            for msg in conversation_history[-history_limit:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            logger.info(f"Including {len(conversation_history[-history_limit:])} messages from history")

        # Add current question
        messages.append({
            "role": "user",
            "content": user_question
        })

        # Call OpenAI API with retry logic
        client = OpenAI(api_key=Config.OPENAI_API_KEY)

        max_retries = AppConfig.LLM_MAX_RETRIES
        answer_text = None

        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=Config.OPENAI_MODEL,
                    messages=messages,
                    max_completion_tokens=Config.MAX_COMPLETION_TOKENS,
                    timeout=Config.OPENAI_TIMEOUT
                )

                # Extract answer
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    choice = response.choices[0]
                    if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                        answer_text = choice.message.content.strip()
                    elif hasattr(choice, 'text'):
                        answer_text = choice.text.strip()
                    else:
                        answer_text = str(choice).strip()
                else:
                    answer_text = str(response).strip()

                # Success - break retry loop
                break

            except OpenAIError as e:
                logger.error(f"OpenAI API error (attempt {attempt + 1}/{max_retries + 1}): {e}")

                if attempt == max_retries:
                    # Final attempt failed - use fallback
                    logger.error("All OpenAI API attempts failed, using fallback response")
                    answer_text = generate_llm_failure_fallback(context, user_question, intents, conversation_history)
                    break

                # Wait before retry (exponential backoff)
                time.sleep(2 ** attempt)

            except Exception as e:
                logger.error(f"Unexpected error calling OpenAI API: {e}", exc_info=True)

                if attempt == max_retries:
                    # Final attempt failed - use fallback
                    answer_text = generate_llm_failure_fallback(context, user_question, intents, conversation_history)
                    break

                time.sleep(2 ** attempt)

        metrics.llm_generation_time = time.time() - generation_start
        metrics.total_time = time.time() - pipeline_start

        # Parse ANSWER and NUDGE from LLM response
        answer_text, nudge_text = parse_answer_nudge(answer_text)
        if nudge_text:
            logger.info(f"Parsed nudge: {nudge_text[:AppConfig.NUDGE_LOG_PREVIEW_LENGTH]}...")

        # Generate delayed nudge for re-engagement after ~60 seconds
        delayed_nudge_text = None
        try:
            from shared.delayed_nudge import generate_delayed_nudge
            delayed_nudge_text = generate_delayed_nudge(
                conversation_history=conversation_history,
                last_answer=answer_text,
                last_nudge=nudge_text
            )
            if delayed_nudge_text:
                logger.info(f"Generated delayed nudge: {delayed_nudge_text[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to generate delayed nudge: {e}")

        # Estimate confidence
        top_score = top_hits[0].get("hybrid_score", 0.0) if top_hits else 0.0
        confidence = min(top_score, 1.0)

        logger.info("=" * 60)
        logger.info("RAG PIPELINE COMPLETE")
        logger.info(f"Total time: {metrics.total_time:.2f}s")
        logger.info(f"Retrieved: {metrics.retrieved_count}, Used: {metrics.chunks_used}")
        logger.info(f"Confidence: {confidence:.2f}")
        logger.info("=" * 60)

        return PipelineResult(
            answer=answer_text,
            sources=sources,
            metrics=metrics,
            filters=filters,
            rewritten_queries=rewritten_queries,
            retrieved_chunks=[hit.get("payload", {}).get("text", "")[:AppConfig.CHUNK_PREVIEW_LENGTH] for hit in top_hits],
            intents=intents,
            confidence_score=confidence,
            search_time_ms=metrics.search_time_ms,
            llm_time_ms=metrics.llm_generation_time * 1000,
            nudge=nudge_text,
            delayed_nudge=delayed_nudge_text
        )

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        metrics.total_time = time.time() - pipeline_start

        return PipelineResult(
            answer=f"An error occurred while processing your question: {str(e)}",
            sources=[],
            metrics=metrics,
            filters={},
            rewritten_queries=[],
            retrieved_chunks=[],
            intents=[],
            confidence_score=0.0,
            search_time_ms=metrics.search_time_ms,
            llm_time_ms=metrics.llm_generation_time * 1000
        )


# ========================
# CLI Entry Point
# ========================
def main():
    """Command-line interface for testing"""
    import sys

    parser = argparse.ArgumentParser(description="RAG Pipeline Query Tool")
    parser.add_argument(
        "question",
        nargs="+",
        help="Question to answer"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=Config.TOP_K,
        help=f"Number of chunks to use (default: {Config.TOP_K})"
    )
    parser.add_argument(
        "--llm-rewrite",
        action="store_true",
        help="Enable LLM-based query rewriting"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Join question parts
    question = " ".join(args.question)

    try:
        # Run pipeline
        result = answer_question(
            question,
            top_k=args.top_k,
            enable_llm_rewrite=args.llm_rewrite
        )

        if args.json:
            # Output as JSON
            print(json.dumps(result.to_dict(), indent=2))
        else:
            # Human-readable output
            print("\n" + "=" * 60)
            print("ANSWER")
            print("=" * 60)
            print(result.answer)
            print("\n" + "=" * 60)
            print("SOURCES")
            print("=" * 60)
            for i, source in enumerate(result.sources, 1):
                print(f"{i}. {source}")

            print("\n" + "=" * 60)
            print("METRICS")
            print("=" * 60)
            print(f"Total time: {result.metrics.total_time:.2f}s")
            print(f"  - Preprocessing: {result.metrics.preprocessing_time:.2f}s")
            print(f"  - LLM rewrite: {result.metrics.llm_rewrite_time:.2f}s")
            print(f"  - Retrieval: {result.metrics.retrieval_time:.2f}s")
            print(f"  - Reranking: {result.metrics.rerank_time:.2f}s")
            print(f"  - Generation: {result.metrics.llm_generation_time:.2f}s")
            print(f"Retrieved: {result.metrics.retrieved_count}")
            print(f"Chunks used: {result.metrics.chunks_used}")
            print(f"Context tokens: {result.metrics.context_tokens}")
            print(f"Confidence: {result.confidence_score:.2f}")

            if result.filters:
                print(f"Filters applied: {result.filters}")

            print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
