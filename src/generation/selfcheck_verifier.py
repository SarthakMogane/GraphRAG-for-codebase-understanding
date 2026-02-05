"""
SelfCheckGPT - Complete Implementation
Zero-resource hallucination detection through sampling consistency
"""

import sys
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
import numpy as np
import nltk
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_client import LangChainClient


class APIRateLimiter:
    def __init__(self, max_calls=5, period=60):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    async def acquire(self):
        now = time.time()
        # Remove calls older than the period
        self.calls = [t for t in self.calls if now - t < self.period]
        
        if len(self.calls) >= self.max_calls:
            wait_time = self.period - (now - self.calls[0]) + 1
            logger.warning(f"Rate limit reached. Waiting {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)
            # Re-clean after wait
            self.calls = [t for t in self.calls if time.time() - t < self.period]
            
        self.calls.append(time.time())

# Global limiter instance
limiter = APIRateLimiter(max_calls=4, period=65) # Conservative: 4 calls per 65s

from pydantic import BaseModel, Field
from typing import List, Optional

class SentenceVerification(BaseModel):
    """
    Pydantic Model: Validates LLM output automatically.
    """
    sentence: str
    consistency_score: float = Field(..., ge=0.0, le=1.0) # Enforce 0-1 range
    is_hallucination: bool
    supporting_samples: List[int] = Field(default_factory=list)
    evidence: str = Field(alias="explanation") # Magic: Auto-maps 'explanation' -> 'evidence'

    class Config:
        populate_by_name = True





class SelfCheckGPT:
    """
    Complete SelfCheckGPT implementation
    
    Method: Generate N alternative answers with high temperature,
    then check if original answer's claims appear in samples.
    """
    
    def __init__(
        self,
        llm_client: LangChainClient,
        n_samples: int = 1,
        temperature: float = 1.0,
        consistency_threshold: float = 0.5
    ):
        self.llm = llm_client
        self.n_samples = n_samples
        self.temperature = temperature
        self.threshold = consistency_threshold
        
        logger.info(f"SelfCheckGPT initialized (n={n_samples}, T={temperature})")
    
    async def verify(
        self,
        query: str,
        context: str,
        answer: str
    ) -> List[SentenceVerification]:
        """
        Standard verification: Generates samples internally then verifies.
        Optimized to generate samples in PARALLEL.
        """
        logger.info("Running SelfCheckGPT verification")
        
        # Step 1: Generate N alternative samples (Parallelized)
        samples = await self._generate_samples_parallel(query, context)
        logger.info(f"Generated {len(samples)} alternative samples")
        
        # Step 2: Verify using these samples
        return await self.verify_with_samples(query, context, answer, samples)

    async def verify_with_samples(
        self, 
        query: str, 
        context: str, 
        answer: str, 
        samples: List[str]
    ) -> List[SentenceVerification]:
        """
        Verify answer using PRE-GENERATED samples.
        Batched Verification: Verifies ALL sentences in 1 API call.
        """
        # Step 1: Split answer into sentences
        sentences = self._split_sentences(answer)
        logger.info(f"Batch Verifying {len(sentences)} sentences against {len(samples)} samples")
        
        print("sentences from answer",sentences)
        print("samples generated:---",samples)
        # 1. Acquire Rate Limit Slot (Wait if needed)
        await limiter.acquire()  #--------problem with multiple workers------------------------------

        print("------moving towards return to perform verify batch ----------")
        # 2. Run Batch Verification
        return await self._verify_batch(sentences, samples)
    
       # the below code is for if we want to pass those each sentence as each api request then 
    #    then we need to send this parrallely using semaphore

        # === RATE LIMIT PROTECTION ===
        # Create a Semaphore that allows only 5 concurrent requests
        # Adjust '5' based on your API tier (Tier 1 usually allows ~60 RPM)
        # concurrency_limit = 5
        # sem = asyncio.Semaphore(concurrency_limit)


        # # Step 2: Check each sentence
        # # We process sentences sequentially to avoid hitting rate limits 
        # # (verification is lighter than sampling, so sequential is usually fine here)
        # verifications = []
        # for i, sentence in enumerate(sentences):
        #     verification = await self._verify_sentence(
        #         sentence, 
        #         samples, 
        #         i, 
        #         len(sentences)
        #     )
        #     verifications.append(verification)
        

        # # Helper wrapper to enforce the limit
        # async def bounded_verify(sentence, idx):
        #     # The 'async with' block waits here until a "slot" is open
        #     async with sem:
        #         return await self._verify_sentence(
        #             sentence, samples, idx, len(sentences)
        #         )
            
        # # Create tasks using the bounded wrapper
        # check_tasks = []
        # for i, sentence in enumerate(sentences):
        #     check_tasks.append(bounded_verify(sentence, i))
        
        # print(type(check_tasks))
        # print("check task are :",check_tasks)
        # # Run them. They will start all at once, but only 5 will execute
        # # the LLM call effectively at any given moment.
        # verifications = await asyncio.gather(*check_tasks)


        # # Log summary
        # hallucinated = sum(1 for v in verifications if v.is_hallucination)
        # logger.info(f"Verification complete: {hallucinated}/{len(sentences)} flagged")
        
        # return verifications

    async def _generate_samples_parallel(self, query: str, context: str) -> List[str]:
        """Generate N alternative answers in PARALLEL using asyncio.gather"""
        
        system_prompt = "You are a code analysis expert. Answer based on the provided context."
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nProvide a concise answer based only on the context."
        
        # Create tasks for parallel execution
        tasks = []
        for i in range(self.n_samples):
            # We use ainvoke if available (preferred), else generate
            if hasattr(self.llm, "ainvoke"):
                print("it hasattr",i)
                tasks.append(self.llm.ainvoke(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    config={"tags": ["VERIFICATION_SAMPLE_GEN"],} 
                ))
            else:
                # Fallback to generate
                tasks.append(self.llm.generate(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature
                ))
        print("outside of loop")
        print("the task list contain :--",tasks)
        # Run all generation tasks concurrently
        try:
            print("runing coroutine objects using gather.....")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            print("result from asyncio.gather :--\n",results)
            
            samples = []
            for i, res in enumerate(results):
                print("loop no : ",i)
                print("result got for this loop:-- \n",res)
                if isinstance(res, Exception):
                    logger.error(f"Sample {i+1} failed: {res}")
                else:
                    # Handle both LangChain Message objects and string responses
                    content = res.content if hasattr(res, 'content') else str(res)
                    samples.append(content)

            print("sample we got ",samples)
                    
                    
            return samples
        
            
        except Exception as e:
            logger.error(f"Parallel sampling crashed: {e}")
            return []

    async def _verify_batch(
            self, sentences: List[str], samples: List[str]
        ) -> List[SentenceVerification]:
            """
            Verify multiple sentences in a SINGLE LLM call to save requests.
            """
            # Prepare the numbered list of sentences
            print("Debug : inside verify_batch :--")

            sentences_text = ""
            for i, s in enumerate(sentences):
                sentences_text += f"<sentence id='{i}'>{s}</sentence>\n"

            # Prepare samples
            samples_text = ""
            for i, s in enumerate(samples):
                samples_text += f"<sample id='{i}'>{s}</sample>\n"

            print("sentences text  from input sentences:---\n ",sentences_text)
            print("samples text from sample for sending prompt \n:---",samples_text)
            # Prompt asking for a LIST of results
            prompt = f"""You are a fact-checking assistant.

    <task>
    Verify if the information in EACH <sentence> is supported by the <reference_samples>.
    </task>

    <input_sentences>
    {sentences_text}
    </input_sentences>

    <reference_samples>
    {samples_text}
    </reference_samples>

    <instructions>
    Return a JSON object with a list "verifications".
    Order must match the input sentences.

    JSON Schema:
    {{
    "verifications": [
        {{
        "sentence_id": 0,
        "consistency_score": 0.0 to 1.0,
        "explanation": "reasoning"
        }},
        ...
    ]
    }}
    </instructions>
    """
            
            try:
                # We use the generate_json method you already have
                result = await self.llm.generate_json(
                    prompt=prompt,
                    schema={
                        "type": "object",
                        "properties": {
                            "verifications": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "sentence_id": {"type": "integer"},
                                        "consistency_score": {"type": "number"},
                                        "explanation": {"type": "string"}
                                    },
                                    "required": ["sentence_id", "consistency_score", "explanation"]
                                }
                            }
                        },
                        "required": ["verifications"]
                    },
                    temperature=0.0
                )
                
                # Parse results back into objects
                verifications = []
                results_map = {item['sentence_id']: item for item in result['verifications']}
                
                for i, sent in enumerate(sentences):
                    res = results_map.get(i, {"consistency_score": 0.5, "explanation": "Missing"})
                    score = res['consistency_score']
                    is_hallucination = score < self.threshold
                    
                    verifications.append(SentenceVerification(
                        sentence=sent,
                        consistency_score=score,
                        is_hallucination=is_hallucination,
                        supporting_samples=[],
                        evidence=res['explanation']
                    ))
          
                return verifications

            except Exception as e:
                logger.error(f"Batch verification failed: {e}")
                # Fallback: Mark everything as unverified but don't crash
                return [
                    SentenceVerification(s, 0.0, True, "Verification Error") 
                    for s in sentences
                ]
            
    # the below is for single sentence approch for parallely executing each sentence but since we have rate limit we are using batch function 
    async def _verify_sentence(
        self,
        sentence: str,
        samples: List[str],
        sentence_idx: int,
        total_sentences: int
    ) -> SentenceVerification:
        """
        Verify a single sentence against samples
        """
        
        # Build verification prompt
        # Modified Verification Prompt for "Paragraph Mode"
        verification_prompt = f"""You are an expert fact-checking assistant.

        <task>
        Verify if the *factual claims* in the <target_sentence> are consistent with the <reference_samples>.
        </task>

        <target_sentence>
        {sentence}
        </target_sentence>

        <reference_samples>
        """
        for i, sample in enumerate(samples, 1):
            verification_prompt += f"<sample id='{i}'>\n{sample}\n</sample>\n"
        
        verification_prompt += """
        </reference_samples>

        <critical_rules>
        1. **Resolve Pronouns:** The target sentence is part of a paragraph. If it uses pronouns like "it", "this", or "they", infer the subject based on the context found in the <reference_samples>. Do not penalize for missing context if the facts match.
        2. **Ignore Transitions:** Ignore stylistic transition words (e.g., "First", "Next", "Finally", "Additionally") in the target sentence. Focus only on the core information.
        3. **Factual Equivalence:** The wording does not need to be identical. If the meaning is the same, it is consistent.
        </critical_rules>

        <scoring_guide>
        - 1.0: The claim is explicitly supported by the samples (even if phrased differently).
        - 0.5: The claim is implied or partially supported.
        - 0.0: The claim is contradicted or completely absent from the samples.
        </scoring_guide>

        <instructions>
        Respond with JSON:
        {
        "appears_in_samples": [list of integer ids],
        "consistency_score": 0.0 to 1.0,
        "explanation": "Brief reasoning. If you resolved a pronoun (e.g., 'it' -> 'fetch_user_data'), mention that."
        }
        </instructions>
        """
        
        try:
            # Note: Checking is usually fast, so we assume generate_json handles it
            result = await self.llm.generate_json(
                prompt=verification_prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "appears_in_samples": {"type": "array", "items": {"type": "integer"}},
                        "consistency_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "explanation": {"type": "string"}
                    },
                    "required": ["appears_in_samples", "consistency_score", "explanation"]
                },
                temperature=0.0 # Strict for verification
            )
            
            consistency_score = result['consistency_score']
            supporting_samples = result['appears_in_samples']
            evidence = result['explanation']
            
        except Exception as e:
            logger.warning(f"Verification failed for sentence {sentence_idx+1}: {e}")
            # Fallback: simple heuristic
            consistency_score = self._heuristic_check(sentence, samples)
            supporting_samples = []
            evidence = "Fallback heuristic check"
        
        # Determine if hallucination
        is_hallucination = consistency_score < self.threshold
        
        return SentenceVerification(
            sentence=sentence,
            consistency_score=consistency_score,
            is_hallucination=is_hallucination,
            supporting_samples=supporting_samples,
            evidence=evidence
        )
    
    def _heuristic_check(self, sentence: str, samples: List[str]) -> float:
        """Fallback heuristic: count how many samples contain key terms"""
        if not samples:
            return 0.5
        words = sentence.lower().split()
        key_terms = [w.strip('.,!?;:') for w in words if len(w) > 4]
        
        if not key_terms:
            return 0.5
        
        matches = 0
        for sample in samples:
            sample_lower = sample.lower()
            if any(term in sample_lower for term in key_terms):
                matches += 1
        
        return matches / len(samples)
    

    def _split_sentences(self, text: str) -> List[str]:
        """
        Production-Grade Sentence Splitting using NLTK.
        Handles abbreviations (e.g., "Mr.", "Fig. 1", "U.S.A.") correctly.
        """
        text = text.strip()
        if not text:
            return []
            
        try:
            # standard NLTK tokenizer
            return nltk.sent_tokenize(text)
            
        except LookupError:
            # Auto-download 'punkt' model if missing (First run safety)
            logger.info("Downloading NLTK 'punkt' tokenizer model...")
            try:
                nltk.download('punkt')
                nltk.download('punkt_tab') # Required for newer NLTK versions
                return nltk.sent_tokenize(text)
            except Exception as e:
                logger.error(f"NLTK download failed: {e}. Falling back to Regex.")
                # Fallback to Regex if internet/download fails
                import re
                sentences = re.split(r'(?<=[.!?])\s+', text)
                return [s.strip() for s in sentences if s.strip()]
    
    def get_verification_summary(self, verifications: List[SentenceVerification]) -> Dict:
        """Get summary statistics"""
        total = len(verifications)
        hallucinated = sum(1 for v in verifications if v.is_hallucination)
        scores = [v.consistency_score for v in verifications]
        
        return {
            'total_sentences': total,
            'hallucinated_count': hallucinated,
            'hallucination_rate': hallucinated / total if total > 0 else 0,
            'avg_consistency': np.mean(scores) if scores else 0,
            'verified': hallucinated == 0
        }
    
    def format_verified_answer(
        self,
        answer: str,
        verifications: List[SentenceVerification],
        include_scores: bool = False
    ) -> str:
        """
        Format answer with verification indicators (Badges)
        Matches the sentences in 'answer' with 'verifications'
        """
        sentences = self._split_sentences(answer)
        
        # Handle mismatch case (if answer changed/reformatted)
        if len(sentences) != len(verifications):
            logger.warning("Sentence count mismatch in formatting. Returning raw answer.")
            return answer
            
        formatted = []
        for sentence, verification in zip(sentences, verifications):
            if verification.is_hallucination:
                # Warning badge
                badge = " ⚠️[Unverified]"
                formatted.append(f"{sentence}{badge}")
            else:
                # Success badge
                badge = " ✅"
                formatted.append(f"{sentence}{badge}")
        
        return ' '.join(formatted)

    def merge_citations_and_verification(
        self,
        raw_answer: str,
        verifications: List[SentenceVerification],
        citations: Dict, # From CitationExtractor
        graph
    ) -> str:
        """
        Production Assembly: Merges Text + Citations + Verification Badges.
        Ensures perfect alignment of all layers.
        """
        sentences = self._split_sentences(raw_answer)
        
        # Safety Check: Mismatched lengths implies splitting logic differed
        if len(sentences) != len(verifications):
            logger.warning("Formatting mismatch. Returning simple cited answer.")
            # Fallback: Just return citations without badges to avoid data corruption
            return self._fallback_merge(sentences, citations, graph)

        formatted_sentences = []
        
        for i, (sentence, verification) in enumerate(zip(sentences, verifications)):
            parts = [sentence.strip()]
            
            # 1. Append Citations (if any)
            if i in citations:
                entities = citations[i]['entities']
                names = [graph.nodes[nid].get('name', nid) for nid in entities[:3]]

                new_citations = []
                for name in names:
                    if f"[{name}]" not in sentence and f"[[{name}]]" not in sentence:
                        new_citations.append(name)
                
                if new_citations:
                    parts.append(f"[{', '.join(new_citations)}]")

            # 2. Append Verification Badge
            if verification.is_hallucination:
                parts.append("⚠️") # Warning
            else:
                parts.append("✅") # Success
            
            formatted_sentences.append(" ".join(parts))
            
        return " ".join(formatted_sentences)

    def _fallback_merge(self, sentences, citations, graph) -> str:
        """Helper to merge just citations if verification fails alignment"""
        formatted = []
        for i, s in enumerate(sentences):
            if i in citations:
                names = [graph.nodes[e].get('name', e) for e in citations[i]['entities'][:3]]
                formatted.append(f"{s} [{', '.join(names)}]")
            else:
                formatted.append(s)
        return " ".join(formatted)

class CitationExtractor:
    """Extract citations and link to source code entities"""
    
    def __init__(self, graph):
        self.graph = graph
        # Build entity name index (lowercase -> node_id)
        self.entity_index = {}
        for node_id, attrs in graph.nodes(data=True):
            name = attrs.get('name', '')
            if name:
                self.entity_index[name.lower()] = node_id
        
        logger.info(f"Citation extractor initialized ({len(self.entity_index)} entities)")
    
    def extract_citations(self, answer: str, context_nodes: List[str]) -> Dict:
        """Identify which context nodes are mentioned in the answer"""
        citations = {}
        sentences = self._split_sentences(answer)
        
        for i, sentence in enumerate(sentences):
            cited = self._find_entities_in_sentence(sentence)
            # Filter: only cite entities that were actually in the context
            valid_citations = [e for e in cited if e in context_nodes]
            
            if valid_citations:
                citations[i] = {
                    'sentence': sentence,
                    'entities': valid_citations
                }
        return citations
    
    def format_with_citations(self, answer: str, citations: Dict, graph) -> str:
        """Add [NodeName] links to the text"""
        sentences = self._split_sentences(answer)
        formatted = []
        
        for i, sentence in enumerate(sentences):
            if i in citations:
                entities = citations[i]['entities']
                # Get clean names
                names = [graph.nodes[nid].get('name', nid) for nid in entities[:3]]
                # Format: "Sentence text. [Node1, Node2]"
                formatted.append(f"{sentence} [{', '.join(names)}]")
            else:
                formatted.append(sentence)
                
        return ' '.join(formatted)
        
    def _find_entities_in_sentence(self, sentence: str) -> List[str]:
        sentence_lower = sentence.lower()
        found = []
        # Simple string matching (can be improved with fuzzy match)
        for name, node_id in self.entity_index.items():
            if len(name) > 3 and name in sentence_lower: # Avoid short noise matches
                found.append(node_id)
        return list(set(found)) # Deduplicate
    
    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]








# """
# SelfCheckGPT - Complete Implementation
# Zero-resource hallucination detection through sampling consistency
# """

# import sys
# from pathlib import Path
# from typing import List, Dict, Tuple
# import re
# from dataclasses import dataclass
# from loguru import logger
# import numpy as np

# sys.path.insert(0, str(Path(__file__).parent.parent))

# from utils.llm_client import LangChainClient


# @dataclass
# class SentenceVerification:
#     """Verification result for a single sentence"""
#     sentence: str
#     consistency_score: float  # 0-1, higher = more consistent
#     is_hallucination: bool  # True if likely hallucinated
#     supporting_samples: List[int]  # Indices of supporting samples
#     evidence: str  # Brief explanation


# class SelfCheckGPT:
#     """
#     Complete SelfCheckGPT implementation
    
#     Method: Generate N alternative answers with high temperature,
#     then check if original answer's claims appear in samples.
    
#     Intuition: If model is certain (not hallucinating), all samples
#     will contain similar information. If hallucinating, samples diverge.
#     """
    
#     def __init__(
#         self,
#         llm_client: LangChainClient,
#         n_samples: int = 2,
#         temperature: float = 1.0,
#         consistency_threshold: float = 0.5
#     ):
#         """
#         Initialize SelfCheckGPT verifier
        
#         Args:
#             llm_client: LLM client for generation
#             n_samples: Number of alternative samples to generate
#             temperature: Sampling temperature (higher = more random)
#             consistency_threshold: Threshold for hallucination detection
#         """
#         self.llm = llm_client
#         self.n_samples = n_samples
#         self.temperature = temperature
#         self.threshold = consistency_threshold
        
#         logger.info(f"SelfCheckGPT initialized (n={n_samples}, T={temperature})")
    
#     async def verify(
#         self,
#         query: str,
#         context: str,
#         answer: str
#     ) -> List[SentenceVerification]:
#         """
#         Verify answer for hallucinations
        
#         Args:
#             query: Original query
#             context: Context used for generation
#             answer: Generated answer to verify
        
#         Returns:
#             List of SentenceVerification for each sentence
#         """
#         logger.info("Running SelfCheckGPT verification")
        
#         # Step 1: Split answer into sentences
#         sentences = self._split_sentences(answer)
#         logger.info(f"Verifying {len(sentences)} sentences")
        
#         # Step 2: Generate N alternative samples
#         samples = await self._generate_samples(query, context)
#         logger.info(f"Generated {len(samples)} alternative samples")
        
#         # Step 3: Check each sentence
#         verifications = []
#         for i, sentence in enumerate(sentences):
#             verification = self._verify_sentence(
#                 sentence, 
#                 samples, 
#                 i, 
#                 len(sentences)
#             )
#             verifications.append(verification)
        
#         # Log summary
#         hallucinated = sum(1 for v in verifications if v.is_hallucination)
#         logger.info(f"Verification complete: {hallucinated}/{len(sentences)} "
#                    f"potential hallucinations")
        
#         return verifications
    
#     async def _generate_samples(self, query: str, context: str) -> List[str]:
#         """Generate N alternative answers with high temperature"""
        
#         samples = []
        
#         system_prompt = "You are a code analysis expert. Answer based on the provided context."
        
#         user_prompt = f"""Context:
# {context}

# Question: {query}

# Provide a concise answer based only on the context."""
        
#         for i in range(self.n_samples):
#             try:
#                 response = await self.llm.generate(
#                     messages=[
#                         {"role": "system", "content": system_prompt},
#                         {"role": "user", "content": user_prompt}
#                     ],
#                     temperature=self.temperature,  # High for diversity
#                     # max_tokens=500
#                 )
#                 samples.append(response.content)
                
#             except Exception as e:
#                 logger.error(f"Sample {i+1} generation failed: {e}")
        
#         return samples
    
#     def _verify_sentence(
#         self,
#         sentence: str,
#         samples: List[str],
#         sentence_idx: int,
#         total_sentences: int
#     ) -> SentenceVerification:
#         """
#         Verify a single sentence against samples
        
#         Uses LLM to check if sentence information appears in samples
#         """
        
#         # Build verification prompt
#         verification_prompt = f"""You are verifying factual consistency.

# **Sentence to verify:**
# {sentence}

# **Reference samples (generated independently):**
# """
        
#         for i, sample in enumerate(samples, 1):
#             verification_prompt += f"\nSample {i}:\n{sample}\n"
        
#         verification_prompt += """
# **Task:**
# Does the information in the sentence appear in the reference samples?

# Respond with JSON:
# {
#   "appears_in_samples": [list of sample numbers, 1-5],
#   "consistency_score": 0.0 to 1.0,
#   "explanation": "brief reasoning"
# }

# Score guide:
# - 1.0: Information clearly in all/most samples
# - 0.5: Partially mentioned or implied
# - 0.0: Not mentioned or contradicted
# """
        
#         try:
#             result = self.llm.generate_json(
#                 prompt=verification_prompt,
#                 schema={
#                     "type": "object",
#                     "properties": {
#                         "appears_in_samples": {
#                             "type": "array",
#                             "items": {"type": "integer"}
#                         },
#                         "consistency_score": {
#                             "type": "number",
#                             "minimum": 0,
#                             "maximum": 1
#                         },
#                         "explanation": {"type": "string"}
#                     },
#                     "required": ["appears_in_samples", "consistency_score", "explanation"]
#                 },
#                 temperature=0.3  # Lower for consistent verification
#             )
            
#             consistency_score = result['consistency_score']
#             supporting_samples = result['appears_in_samples']
#             evidence = result['explanation']
            
#         except Exception as e:
#             logger.warning(f"Verification failed for sentence {sentence_idx+1}: {e}")
#             # Fallback: simple heuristic
#             consistency_score = self._heuristic_check(sentence, samples)
#             supporting_samples = []
#             evidence = "Fallback heuristic check"
        
#         # Determine if hallucination
#         is_hallucination = consistency_score < self.threshold
        
#         return SentenceVerification(
#             sentence=sentence,
#             consistency_score=consistency_score,
#             is_hallucination=is_hallucination,
#             supporting_samples=supporting_samples,
#             evidence=evidence
#         )
    
#     def _heuristic_check(self, sentence: str, samples: List[str]) -> float:
#         """
#         Fallback heuristic: count how many samples contain key terms
#         """

#         if not samples:
#             logger.warning("No samples available for heuristic check. Assuming uncertain (0.5).")
#             return 0.5
#         # Extract key terms (simple: words longer than 4 chars)
#         words = sentence.lower().split()
#         key_terms = [w.strip('.,!?;:') for w in words if len(w) > 4]
        
#         if not key_terms:
#             return 0.5
        
#         # Count samples containing key terms
#         matches = 0
#         for sample in samples:
#             sample_lower = sample.lower()
#             if any(term in sample_lower for term in key_terms):
#                 matches += 1
        
#         return matches / len(samples)
    
#     def _split_sentences(self, text: str) -> List[str]:
#         """Split text into sentences"""
#         # Simple sentence splitter
#         # In production, use spacy or nltk
#         sentences = re.split(r'(?<=[.!?])\s+', text)
#         sentences = [s.strip() for s in sentences if s.strip()]
#         return sentences
    
#     def get_verification_summary(
#         self,
#         verifications: List[SentenceVerification]
#     ) -> Dict:
#         """Get summary statistics of verification"""
        
#         total = len(verifications)
#         hallucinated = sum(1 for v in verifications if v.is_hallucination)
        
#         scores = [v.consistency_score for v in verifications]
        
#         return {
#             'total_sentences': total,
#             'hallucinated_count': hallucinated,
#             'hallucination_rate': hallucinated / total if total > 0 else 0,
#             'avg_consistency': np.mean(scores) if scores else 0,
#             'min_consistency': np.min(scores) if scores else 0,
#             'verified': hallucinated == 0
#         }
    
#     def format_verified_answer(
#         self,
#         answer: str,
#         verifications: List[SentenceVerification],
#         include_scores: bool = False
#     ) -> str:
#         """
#         Format answer with verification indicators
        
#         Args:
#             answer: Original answer
#             verifications: Verification results
#             include_scores: Include consistency scores
        
#         Returns:
#             Formatted answer with indicators
#         """
#         sentences = self._split_sentences(answer)
        
#         formatted = []
#         for sentence, verification in zip(sentences, verifications):
#             if verification.is_hallucination:
#                 # Mark as potentially unreliable
#                 formatted.append(f"⚠️ {sentence}")
#                 if include_scores:
#                     formatted.append(f"   (Consistency: {verification.consistency_score:.2f})")
#             else:
#                 formatted.append(f"✓ {sentence}")
#                 if include_scores:
#                     formatted.append(f"   (Consistency: {verification.consistency_score:.2f})")
        
#         return '\n'.join(formatted)


# class CitationExtractor:
#     """
#     Extract citations from answer and link to source code
#     """
    
#     def __init__(self, graph):
#         """
#         Initialize citation extractor
        
#         Args:
#             graph: NetworkX graph with code entities
#         """
#         self.graph = graph
        
#         # Build entity name index
#         self.entity_index = {
#             attrs.get('name', '').lower(): node_id
#             for node_id, attrs in graph.nodes(data=True)
#         }
        
#         logger.info(f"Citation extractor initialized ({len(self.entity_index)} entities)")
    
#     def extract_citations(self, answer: str, context_nodes: List[str]) -> Dict:
#         """
#         Extract citations from answer
        
#         Args:
#             answer: Generated answer
#             context_nodes: Node IDs that were in context
        
#         Returns:
#             Dictionary mapping sentences to cited entities
#         """
#         citations = {}
        
#         sentences = self._split_sentences(answer)
        
#         for i, sentence in enumerate(sentences):
#             cited_entities = self._find_entities_in_sentence(sentence)
            
#             # Filter to only entities that were in context
#             cited_entities = [
#                 entity for entity in cited_entities
#                 if entity in context_nodes
#             ]
            
#             if cited_entities:
#                 citations[i] = {
#                     'sentence': sentence,
#                     'entities': cited_entities
#                 }
        
#         return citations
    
#     def _find_entities_in_sentence(self, sentence: str) -> List[str]:
#         """Find entity mentions in sentence"""
        
#         sentence_lower = sentence.lower()
#         mentioned_entities = []
        
#         for entity_name, node_id in self.entity_index.items():
#             if entity_name in sentence_lower:
#                 mentioned_entities.append(node_id)
        
#         return mentioned_entities
    
#     def _split_sentences(self, text: str) -> List[str]:
#         """Split text into sentences"""
#         sentences = re.split(r'(?<=[.!?])\s+', text)
#         return [s.strip() for s in sentences if s.strip()]
    
#     def format_with_citations(
#         self,
#         answer: str,
#         citations: Dict,
#         graph
#     ) -> str:
#         """Format answer with inline citations"""
        
#         sentences = self._split_sentences(answer)
#         formatted = []
        
#         for i, sentence in enumerate(sentences):
#             if i in citations:
#                 entities = citations[i]['entities']
#                 entity_names = [
#                     graph.nodes[node_id].get('name', 'unknown')
#                     for node_id in entities[:3]  # Top 3
#                 ]
#                 citation_str = f" [{', '.join(entity_names)}]"
#                 formatted.append(sentence + citation_str)
#             else:
#                 formatted.append(sentence)
        
#         return ' '.join(formatted)


# # Example usage
# if __name__ == "__main__":
#     from utils.llm_client import LangChainClient
    
#     # Initialize
#     llm = LangChainClient(temperature=0.3)
#     verifier = SelfCheckGPT(llm, n_samples=3)  # 3 for demo
    
#     # Test data
#     query = "How does validation work?"
#     context = """
#     The validate_email function checks email format using regex.
#     The check_user_input function validates all user inputs.
#     """
    
#     answer = """The system validates data through multiple functions. 
#     The validate_email function uses regex patterns to check email format. 
#     The check_user_input function validates all user inputs including email, 
#     phone numbers, and addresses. 
#     The system also uses machine learning for advanced validation."""
    
#     # Verify
#     print("Running SelfCheckGPT verification...\n")
#     verifications = verifier.verify(query, context, answer)
    
#     # Show results
#     for i, v in enumerate(verifications, 1):
#         print(f"{i}. {v.sentence}")
#         print(f"   Consistency: {v.consistency_score:.2f}")
#         print(f"   Hallucination: {v.is_hallucination}")
#         print(f"   Evidence: {v.evidence}\n")
    
#     # Summary
#     summary = verifier.get_verification_summary(verifications)
#     print(f"Summary:")
#     print(f"  Total: {summary['total_sentences']}")
#     print(f"  Hallucinated: {summary['hallucinated_count']}")
#     print(f"  Rate: {summary['hallucination_rate']:.1%}")
#     print(f"  Avg consistency: {summary['avg_consistency']:.2f}")

#     # Show the "Traffic Light" formatted output
#     print("\n--- Verified Output (Traffic Light) ---")
#     formatted_text = verifier.format_verified_answer(
#         answer, 
#         verifications, 
#         include_scores=True
#     )
#     print(formatted_text)