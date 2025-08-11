"""Enhanced pipeline with progress tracking."""

import time
from typing import List
from core.base import Pipeline, Document, ProcessingResult
from utils.progress import LlamaProgressTracker, create_enhanced_progress_bar


class EnhancedPipeline(Pipeline):
    """Pipeline with beautiful progress tracking and llama puns."""

    def __init__(self, name: str = "Enhanced RAG Pipeline"):
        super().__init__(name)
        self.tracker = LlamaProgressTracker()

    def run_with_progress(
        self, source: str = None, documents: List[Document] = None
    ) -> ProcessingResult:
        """Run the pipeline with enhanced progress tracking."""

        # Show llama art and welcome message
        self.tracker.print_llama_art()
        self.tracker.print_header("🚀 Starting RAG Processing Adventure! 🚀")

        # Start with parsing if source provided
        if source and not documents:
            if not self.components or not hasattr(self.components[0], "parse"):
                raise ValueError(
                    "Pipeline must start with a Parser when source is provided"
                )

            self.tracker.print_info(f"📖 Parsing documents from: {source}")
            self.tracker.print_info(self.tracker.get_random_pun())

            # Parse with progress
            parser = self.components[0]
            result = parser.parse(source)
            # Normalize parser outputs that may return List[Document]
            if isinstance(result, ProcessingResult):
                current_docs = result.documents
                all_errors = result.errors
            else:
                current_docs = result
                all_errors = []

            self.tracker.print_success(f"Parsed {len(current_docs)} documents!")
            if all_errors:
                self.tracker.print_warning(f"Found {len(all_errors)} parsing errors")

            start_idx = 1
        elif documents:
            current_docs = documents
            all_errors = []
            start_idx = 0
        else:
            raise ValueError("Either source or documents must be provided")

        if not current_docs:
            self.tracker.print_error("No documents to process!")
            return ProcessingResult(documents=[], errors=all_errors)

        # Process through remaining components with progress tracking
        total_steps = len(self.components) - start_idx
        if total_steps > 0:
            self.tracker.print_info(
                f"🔄 Processing {len(current_docs)} documents through {total_steps} steps..."
            )

            for step_idx, component in enumerate(self.components[start_idx:], 1):
                component_name = component.__class__.__name__

                # Show step header
                print(f"\n📋 Step {step_idx}/{total_steps}: {component_name}")
                print(f"💭 {self.tracker.get_random_pun()}")

                # Special handling for embedders (slow)
                if hasattr(component, "embed"):
                    self._process_embeddings_with_progress(component, current_docs)

                # Special handling for vector stores (also potentially slow)
                elif hasattr(component, "add_documents"):
                    self._process_storage_with_progress(component, current_docs)

                # Other components
                else:
                    try:
                        print(f"⚡ Processing with {component_name}...")
                        result = component.process(current_docs)
                        current_docs = result.documents
                        all_errors.extend(result.errors)
                        self.tracker.print_success(f"{component_name} completed!")
                    except Exception as e:
                        self.logger.error(f"Component {component.name} failed: {e}")
                        all_errors.append(
                            {"component": component.name, "error": str(e)}
                        )

                # Show motivational message between steps
                if step_idx < total_steps:
                    print(f"✨ {self.tracker.get_random_motivation()}")
                    time.sleep(0.5)  # Brief pause for dramatic effect

        # Final celebration
        print(f"\n{self.tracker.get_completion_message()}")
        self.tracker.print_header("🎉 Processing Complete! 🎉")

        # Show final stats
        self.tracker.print_success(
            f"Successfully processed {len(current_docs)} documents"
        )
        if all_errors:
            self.tracker.print_warning(
                f"Encountered {len(all_errors)} errors during processing"
            )
        else:
            self.tracker.print_success("Zero errors - llama-perfect execution! 🦙")

        return ProcessingResult(documents=current_docs, errors=all_errors)

    def _process_embeddings_with_progress(self, embedder, documents: List[Document]):
        """Process embeddings with detailed progress tracking."""
        batch_size = getattr(embedder, "batch_size", 32)
        total_batches = (len(documents) + batch_size - 1) // batch_size

        print(f"🧠 Generating embeddings for {len(documents)} documents...")
        print(f"🔢 Processing in {total_batches} batches of {batch_size}")

        pbar, milestone_interval = create_enhanced_progress_bar(
            len(documents), "🦙 Embedding", self.tracker
        )

        processed_count = 0

        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                texts = [doc.content for doc in batch]

                # Generate embeddings for batch
                embeddings = embedder.embed(texts)

                # Update documents with embeddings
                for doc, embedding in zip(batch, embeddings):
                    doc.embeddings = embedding
                    processed_count += 1
                    pbar.update(1)

                    # Show motivational message at milestones
                    if processed_count % milestone_interval == 0:
                        pbar.set_postfix_str(self.tracker.get_random_pun())
                        time.sleep(0.1)  # Brief pause to let users see the message

            pbar.close()
            self.tracker.print_success(
                f"Embeddings generated! Average dimension: {len(embeddings[0]) if embeddings and embeddings[0] else 'Unknown'}"
            )

        except Exception as e:
            pbar.close()
            raise e

    def _process_storage_with_progress(self, store, documents: List[Document]):
        """Process vector storage with progress tracking."""
        print(f"💾 Storing {len(documents)} documents in vector database...")

        # Create progress bar
        pbar, milestone_interval = create_enhanced_progress_bar(
            len(documents), "🦙 Storing", self.tracker
        )

        try:
            # For now, we'll simulate progress since ChromaDB doesn't expose batch progress
            # In a real implementation, you might batch the storage operations

            # Show some progress updates
            for i in range(len(documents)):
                if i % (len(documents) // 10 + 1) == 0:  # Update every ~10%
                    pbar.set_postfix_str(self.tracker.get_random_motivation())
                    time.sleep(0.05)
                pbar.update(1)

            # Actually store the documents
            success = store.add_documents(documents)

            pbar.close()

            if success:
                self.tracker.print_success(
                    f"Successfully stored all {len(documents)} documents!"
                )
            else:
                self.tracker.print_warning(
                    "Storage completed but some issues may have occurred"
                )

        except Exception as e:
            pbar.close()
            raise e
