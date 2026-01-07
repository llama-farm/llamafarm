"""
CFFI bindings for llama.cpp C API.

Uses ABI mode to dynamically load the pre-built libllama library.
"""

from __future__ import annotations

import atexit
import logging
from typing import TYPE_CHECKING

import cffi

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# CFFI instance
ffi = cffi.FFI()

# C declarations for llama.cpp API
# Based on llama.h from llama.cpp
LLAMA_CDEF = """
    // Opaque types
    typedef struct llama_model llama_model;
    typedef struct llama_context llama_context;
    typedef struct llama_sampler llama_sampler;
    typedef struct llama_vocab llama_vocab;
    typedef struct llama_memory_i * llama_memory_t;

    typedef int32_t llama_pos;
    typedef int32_t llama_token;
    typedef int32_t llama_seq_id;

    // Token types
    enum llama_token_type {
        LLAMA_TOKEN_TYPE_UNDEFINED    = 0,
        LLAMA_TOKEN_TYPE_NORMAL       = 1,
        LLAMA_TOKEN_TYPE_UNKNOWN      = 2,
        LLAMA_TOKEN_TYPE_CONTROL      = 3,
        LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
        LLAMA_TOKEN_TYPE_UNUSED       = 5,
        LLAMA_TOKEN_TYPE_BYTE         = 6,
    };

    // Vocab types
    enum llama_vocab_type {
        LLAMA_VOCAB_TYPE_NONE = 0,
        LLAMA_VOCAB_TYPE_SPM  = 1,
        LLAMA_VOCAB_TYPE_BPE  = 2,
        LLAMA_VOCAB_TYPE_WPM  = 3,
        LLAMA_VOCAB_TYPE_UGM  = 4,
        LLAMA_VOCAB_TYPE_RWKV = 5,
    };

    // Model file type
    enum llama_ftype {
        LLAMA_FTYPE_ALL_F32              = 0,
        LLAMA_FTYPE_MOSTLY_F16           = 1,
        LLAMA_FTYPE_MOSTLY_Q4_0          = 2,
        LLAMA_FTYPE_MOSTLY_Q4_1          = 3,
        LLAMA_FTYPE_MOSTLY_Q8_0          = 7,
        LLAMA_FTYPE_MOSTLY_Q5_0          = 8,
        LLAMA_FTYPE_MOSTLY_Q5_1          = 9,
        LLAMA_FTYPE_MOSTLY_Q2_K          = 10,
        LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11,
        LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12,
        LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13,
        LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14,
        LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15,
        LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16,
        LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17,
        LLAMA_FTYPE_MOSTLY_Q6_K          = 18,
    };

    // Rope scaling type
    enum llama_rope_scaling_type {
        LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
        LLAMA_ROPE_SCALING_TYPE_NONE        = 0,
        LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1,
        LLAMA_ROPE_SCALING_TYPE_YARN        = 2,
        LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = 2,
    };

    // Pooling type
    enum llama_pooling_type {
        LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
        LLAMA_POOLING_TYPE_NONE = 0,
        LLAMA_POOLING_TYPE_MEAN = 1,
        LLAMA_POOLING_TYPE_CLS  = 2,
        LLAMA_POOLING_TYPE_LAST = 3,
        LLAMA_POOLING_TYPE_RANK = 4,
    };

    // Attention type
    enum llama_attention_type {
        LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1,
        LLAMA_ATTENTION_TYPE_CAUSAL      = 0,
        LLAMA_ATTENTION_TYPE_NON_CAUSAL  = 1,
    };

    // Flash attention type
    enum llama_flash_attn_type {
        LLAMA_FLASH_ATTN_TYPE_AUTO     = -1,
        LLAMA_FLASH_ATTN_TYPE_DISABLED = 0,
        LLAMA_FLASH_ATTN_TYPE_ENABLED  = 1,
    };

    // Split mode enum
    enum llama_split_mode {
        LLAMA_SPLIT_MODE_NONE  = 0,
        LLAMA_SPLIT_MODE_LAYER = 1,
        LLAMA_SPLIT_MODE_ROW   = 2,
    };

    // Model parameters - must match llama.cpp b7376+ layout exactly
    struct llama_model_params {
        void * devices;                      // ggml_backend_dev_t * (NULL = use all)
        void * tensor_buft_overrides;        // const struct llama_model_tensor_buft_override *
        int32_t n_gpu_layers;
        int32_t split_mode;                  // enum llama_split_mode
        int32_t main_gpu;
        const float * tensor_split;
        void * progress_callback;            // llama_progress_callback
        void * progress_callback_user_data;
        void * kv_overrides;                 // const struct llama_model_kv_override *
        bool vocab_only;
        bool use_mmap;
        bool use_mlock;
        bool check_tensors;
        bool use_extra_bufts;
        bool no_host;
    };

    // Context parameters - must match llama.cpp b7376+ layout exactly
    struct llama_context_params {
        uint32_t n_ctx;
        uint32_t n_batch;
        uint32_t n_ubatch;
        uint32_t n_seq_max;
        int32_t n_threads;
        int32_t n_threads_batch;
        int32_t rope_scaling_type;           // enum llama_rope_scaling_type
        int32_t pooling_type;                // enum llama_pooling_type
        int32_t attention_type;              // enum llama_attention_type
        int32_t flash_attn_type;             // enum llama_flash_attn_type
        float rope_freq_base;
        float rope_freq_scale;
        float yarn_ext_factor;
        float yarn_attn_factor;
        float yarn_beta_fast;
        float yarn_beta_slow;
        uint32_t yarn_orig_ctx;
        float defrag_thold;
        void * cb_eval;                      // ggml_backend_sched_eval_callback
        void * cb_eval_user_data;
        int32_t type_k;                      // enum ggml_type
        int32_t type_v;                      // enum ggml_type
        void * abort_callback;               // ggml_abort_callback
        void * abort_callback_data;
        bool embeddings;
        bool offload_kqv;
        bool no_perf;
        bool op_offload;
        bool swa_full;
        bool kv_unified;
    };

    // Batch
    struct llama_batch {
        int32_t n_tokens;
        llama_token * token;
        float * embd;
        llama_pos * pos;
        int32_t * n_seq_id;
        llama_seq_id ** seq_id;
        int8_t * logits;
    };

    // Chat message
    struct llama_chat_message {
        const char * role;
        const char * content;
    };

    // Logging
    enum ggml_log_level {
        GGML_LOG_LEVEL_NONE  = 0,
        GGML_LOG_LEVEL_DEBUG = 1,
        GGML_LOG_LEVEL_INFO  = 2,
        GGML_LOG_LEVEL_WARN  = 3,
        GGML_LOG_LEVEL_ERROR = 4,
        GGML_LOG_LEVEL_CONT  = 5,
    };

    typedef void (*ggml_log_callback)(int level, const char * text, void * user_data);
    void llama_log_set(ggml_log_callback log_callback, void * user_data);

    // Backend initialization
    void llama_backend_init(void);
    void llama_backend_free(void);

    // GGML backend loading (required for llama.cpp b7376+)
    // Must be called after llama_backend_init() to load compute backends (CPU, CUDA, Metal, etc.)
    void ggml_backend_load_all(void);
    void ggml_backend_load_all_from_path(const char * dir_path);

    // NUMA initialization
    void llama_numa_init(int32_t numa);

    // Model loading
    struct llama_model_params llama_model_default_params(void);
    struct llama_model * llama_load_model_from_file(const char * path_model, struct llama_model_params params);
    void llama_free_model(struct llama_model * model);

    // Context
    struct llama_context_params llama_context_default_params(void);
    struct llama_context * llama_new_context_with_model(struct llama_model * model, struct llama_context_params params);
    void llama_free(struct llama_context * ctx);

    // Model info (new API - llama.cpp b7376+)
    const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
    int32_t llama_model_n_ctx_train(const struct llama_model * model);
    int32_t llama_model_n_embd(const struct llama_model * model);
    int32_t llama_model_n_layer(const struct llama_model * model);
    int32_t llama_model_n_head(const struct llama_model * model);

    // Vocab info (new API - llama.cpp b7376+)
    int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab);
    const char * llama_vocab_get_text(const struct llama_vocab * vocab, llama_token token);
    float llama_vocab_get_score(const struct llama_vocab * vocab, llama_token token);
    int32_t llama_vocab_get_attr(const struct llama_vocab * vocab, llama_token token);
    bool llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token);
    bool llama_vocab_is_control(const struct llama_vocab * vocab, llama_token token);

    // Special tokens (new API - llama.cpp b7376+)
    llama_token llama_vocab_bos(const struct llama_vocab * vocab);
    llama_token llama_vocab_eos(const struct llama_vocab * vocab);
    llama_token llama_vocab_eot(const struct llama_vocab * vocab);
    llama_token llama_vocab_sep(const struct llama_vocab * vocab);
    llama_token llama_vocab_nl(const struct llama_vocab * vocab);
    llama_token llama_vocab_pad(const struct llama_vocab * vocab);

    // Context info
    uint32_t llama_n_ctx(const struct llama_context * ctx);
    uint32_t llama_n_batch(const struct llama_context * ctx);
    uint32_t llama_n_ubatch(const struct llama_context * ctx);

    // Tokenization (llama.cpp b7376+ uses vocab instead of model)
    int32_t llama_tokenize(
        const struct llama_vocab * vocab,
        const char * text,
        int32_t text_len,
        llama_token * tokens,
        int32_t n_tokens_max,
        bool add_special,
        bool parse_special
    );

    // Detokenization (llama.cpp b7376+ uses vocab instead of model)
    int32_t llama_token_to_piece(
        const struct llama_vocab * vocab,
        llama_token token,
        char * buf,
        int32_t length,
        int32_t lstrip,
        bool special
    );

    int32_t llama_detokenize(
        const struct llama_vocab * vocab,
        const llama_token * tokens,
        int32_t n_tokens,
        char * text,
        int32_t text_len_max,
        bool remove_special,
        bool unparse_special
    );

    // Chat templates (llama.cpp b7376+ - model param removed)
    int32_t llama_chat_apply_template(
        const char * tmpl,
        const struct llama_chat_message * chat,
        size_t n_msg,
        bool add_ass,
        char * buf,
        int32_t length
    );

    // Batch operations
    struct llama_batch llama_batch_get_one(
        llama_token * tokens,
        int32_t n_tokens
    );

    struct llama_batch llama_batch_init(
        int32_t n_tokens,
        int32_t embd,
        int32_t n_seq_max
    );

    void llama_batch_free(struct llama_batch batch);

    // Memory/KV cache (new API - llama.cpp b7376+)
    llama_memory_t llama_get_memory(const struct llama_context * ctx);
    void llama_memory_clear(llama_memory_t mem, bool data);
    bool llama_memory_seq_rm(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1);
    void llama_memory_seq_cp(llama_memory_t mem, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1);
    void llama_memory_seq_keep(llama_memory_t mem, llama_seq_id seq_id);
    void llama_memory_seq_add(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta);

    // Decoding
    int32_t llama_decode(struct llama_context * ctx, struct llama_batch batch);

    // Logits and embeddings
    float * llama_get_logits(struct llama_context * ctx);
    float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);
    float * llama_get_embeddings(struct llama_context * ctx);
    float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
    float * llama_get_embeddings_seq(struct llama_context * ctx, llama_seq_id seq_id);

    // Sampling - new API
    struct llama_sampler * llama_sampler_chain_init(struct llama_sampler_chain_params params);
    void llama_sampler_chain_add(struct llama_sampler * chain, struct llama_sampler * smpl);
    struct llama_sampler * llama_sampler_chain_get(const struct llama_sampler * chain, int32_t i);
    int32_t llama_sampler_chain_n(const struct llama_sampler * chain);
    void llama_sampler_free(struct llama_sampler * smpl);

    struct llama_sampler_chain_params {
        bool no_perf;
    };

    struct llama_sampler_chain_params llama_sampler_chain_default_params(void);

    // Sampler types
    struct llama_sampler * llama_sampler_init_greedy(void);
    struct llama_sampler * llama_sampler_init_dist(uint32_t seed);
    struct llama_sampler * llama_sampler_init_top_k(int32_t k);
    struct llama_sampler * llama_sampler_init_top_p(float p, size_t min_keep);
    struct llama_sampler * llama_sampler_init_min_p(float p, size_t min_keep);
    struct llama_sampler * llama_sampler_init_temp(float t);
    struct llama_sampler * llama_sampler_init_temp_ext(float t, float delta, float exponent);
    struct llama_sampler * llama_sampler_init_penalties(
        int32_t n_vocab,
        llama_token special_eos_id,
        llama_token linefeed_id,
        int32_t penalty_last_n,
        float penalty_repeat,
        float penalty_freq,
        float penalty_present,
        bool penalize_nl,
        bool ignore_eos
    );

    // Sample from logits
    llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx);
    void llama_sampler_accept(struct llama_sampler * smpl, llama_token token);
    void llama_sampler_reset(struct llama_sampler * smpl);

    // Performance
    struct llama_perf_context_data {
        double t_start_ms;
        double t_load_ms;
        double t_p_eval_ms;
        double t_eval_ms;
        int32_t n_p_eval;
        int32_t n_eval;
    };

    struct llama_perf_context_data llama_perf_context(const struct llama_context * ctx);
    void llama_perf_context_print(const struct llama_context * ctx);
    void llama_perf_context_reset(struct llama_context * ctx);

    // Utility
    void llama_synchronize(struct llama_context * ctx);
"""

# Parse the C declarations
ffi.cdef(LLAMA_CDEF)

# Global library handle
_lib = None


def get_lib():
    """Get the loaded llama.cpp library."""
    global _lib
    if _lib is None:
        _lib = _load_library()
    return _lib


def _preload_ggml_with_global(lib_dir, system):
    """Pre-load libggml with RTLD_GLOBAL so backend registrations are visible.

    On Linux, when libllama.so is loaded, it pulls in libggml.so as a dependency.
    If libggml.so is loaded with RTLD_LOCAL (the default), then backend registrations
    made by ggml_backend_load_all() won't be visible to libllama.so.

    By pre-loading libggml.so with RTLD_GLOBAL before libllama.so, we ensure that
    the backend registry is globally visible.
    """
    import ctypes

    if system == "Darwin":
        # On macOS, everything is usually linked into libllama.dylib
        return

    # Find libggml
    ggml_lib_path = None
    if system == "Windows":
        ggml_lib_path = lib_dir / "ggml.dll"
    elif system == "Linux":
        # Try versioned first, then unversioned
        for pattern in ["libggml.so.0", "libggml.so"]:
            candidate = lib_dir / pattern
            if candidate.exists():
                ggml_lib_path = candidate
                break

    if not ggml_lib_path or not ggml_lib_path.exists():
        logger.debug(f"libggml not found in {lib_dir}, skipping preload")
        return

    try:
        # Platform-specific RTLD flags
        if system == "Linux":
            RTLD_GLOBAL = 0x100
            RTLD_NOW = 0x2
        else:  # Windows
            RTLD_GLOBAL = 0
            RTLD_NOW = 0

        logger.debug(f"Pre-loading {ggml_lib_path.name} with RTLD_GLOBAL...")
        ctypes.CDLL(str(ggml_lib_path), mode=RTLD_GLOBAL | RTLD_NOW)
        logger.info(f"Pre-loaded {ggml_lib_path} with RTLD_GLOBAL")
    except Exception as e:
        logger.warning(f"Failed to pre-load {ggml_lib_path}: {e}")


def _load_library():
    """Load the llama.cpp shared library."""
    import os
    import platform

    from ._binary import get_lib_path

    lib_path = get_lib_path()
    lib_dir = lib_path.parent
    logger.debug(f"Loading llama.cpp library from: {lib_path}")

    system = platform.system()

    # On Windows, prevent OpenMP conflicts between llama.cpp (libomp140) and
    # other libraries like numpy (libiomp5md). This must be set BEFORE loading.
    if system == "Windows":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # On macOS, set GGML_METAL_PATH_RESOURCES so Metal backend can find shaders
    # The ggml-metal.metal shader file must be in the same directory as the library
    if system == "Darwin":
        metal_shader = lib_dir / "ggml-metal.metal"
        if metal_shader.exists():
            os.environ["GGML_METAL_PATH_RESOURCES"] = str(lib_dir)
            logger.debug(f"Set GGML_METAL_PATH_RESOURCES={lib_dir}")
        else:
            logger.warning(
                f"Metal shader not found at {metal_shader}. GPU acceleration may not work."
            )

    # On Windows, add the library directory to DLL search path so that
    # ggml_backend_load_all() can find backend DLLs (ggml-cpu.dll, etc.)
    if system == "Windows":
        # Use add_dll_directory for Python 3.8+ (recommended method)
        try:
            os.add_dll_directory(str(lib_dir))
            logger.debug(f"Added DLL directory: {lib_dir}")
        except AttributeError:
            pass  # Python < 3.8, fall back to PATH modification

        # Also add to PATH as fallback (needed for some DLL loading scenarios)
        path = os.environ.get("PATH", "")
        if str(lib_dir) not in path:
            os.environ["PATH"] = f"{lib_dir};{path}" if path else str(lib_dir)
            logger.debug(f"Added to PATH: {lib_dir}")

    # On Linux, set LD_LIBRARY_PATH to include the library directory
    # so ggml_backend_load_all() can find backend .so files
    if system == "Linux":
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if str(lib_dir) not in ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld_path}" if ld_path else str(lib_dir)
            logger.debug(f"Updated LD_LIBRARY_PATH to include: {lib_dir}")

    # CRITICAL: Pre-load libggml with RTLD_GLOBAL before loading libllama
    # This ensures backend registrations are globally visible when ggml_backend_load_all() is called
    _preload_ggml_with_global(lib_dir, system)

    try:
        lib = ffi.dlopen(str(lib_path))
        logger.info(f"Loaded llama.cpp library: {lib_path}")
        return lib
    except OSError as e:
        raise RuntimeError(
            f"Failed to load llama.cpp library from {lib_path}: {e}"
        ) from e


# Log level mapping from llama.cpp to Python logging
_LLAMA_LOG_LEVELS = {
    0: logging.NOTSET,   # GGML_LOG_LEVEL_NONE
    1: logging.DEBUG,    # GGML_LOG_LEVEL_DEBUG
    2: logging.INFO,     # GGML_LOG_LEVEL_INFO
    3: logging.WARNING,  # GGML_LOG_LEVEL_WARN
    4: logging.ERROR,    # GGML_LOG_LEVEL_ERROR
    5: logging.DEBUG,    # GGML_LOG_LEVEL_CONT (continuation)
}

# Keep reference to callback to prevent garbage collection
_log_callback = None
_log_buffer = ""  # Buffer for continuation lines


def _setup_log_callback(lib):
    """Set up llama.cpp logging to route through Python logging."""
    global _log_callback, _log_buffer

    llama_logger = logging.getLogger("llamafarm_llama.llama_cpp")

    @ffi.callback("void(int, const char *, void *)")
    def log_callback(level, text, user_data):
        global _log_buffer

        if text == ffi.NULL:
            return

        message = ffi.string(text).decode("utf-8", errors="replace")

        # Handle continuation lines (GGML_LOG_LEVEL_CONT = 5)
        if level == 5:
            _log_buffer += message
            return

        # If we have buffered content, prepend it
        if _log_buffer:
            message = _log_buffer + message
            _log_buffer = ""

        # Strip trailing newline if present
        message = message.rstrip("\n")

        # Skip empty messages
        if not message:
            return

        # Get Python log level
        py_level = _LLAMA_LOG_LEVELS.get(level, logging.DEBUG)

        # Log through Python logging system
        llama_logger.log(py_level, message)

    # Store reference to prevent GC
    _log_callback = log_callback

    # Set the callback
    lib.llama_log_set(log_callback, ffi.NULL)
    logger.debug("llama.cpp logging callback installed")

    # Register cleanup to unset callback before Python exits
    # This prevents segfaults when llama.cpp tries to call the callback
    # after Python's interpreter has started shutting down
    def _cleanup_log_callback():
        try:
            lib.llama_log_set(ffi.NULL, ffi.NULL)
        except Exception:
            pass  # Ignore errors during shutdown

    atexit.register(_cleanup_log_callback)


def _load_ggml_backends():
    """Load GGML compute backends (CPU, CUDA, Metal, Vulkan, etc.).

    CRITICAL: Backend loading must happen in the SAME library context as model loading.
    This means we MUST call ggml_backend_load_all() via CFFI (from libllama) since that's
    the same library that will call llama_load_model_from_file().

    If we call via ctypes from a separately-loaded libggml.so, the backend registrations
    happen in a different static data context and aren't visible to libllama.
    """
    import os
    import platform

    from ._binary import get_lib_path

    lib_path = get_lib_path()
    lib_dir = lib_path.parent
    system = platform.system()

    # Log library directory contents for debugging
    logger.info(f"Loading GGML backends from: {lib_dir}")
    try:
        files = list(lib_dir.iterdir())
        logger.debug(f"Library directory contains {len(files)} files")
        for f in sorted(files):
            size = f.stat().st_size if f.is_file() else 0
            ftype = "dir" if f.is_dir() else ("link" if f.is_symlink() else f"{size:,}b")
            logger.debug(f"  {f.name} ({ftype})")
    except Exception as e:
        logger.warning(f"Could not list library directory: {e}")

    # Ensure LD_LIBRARY_PATH is set (for Linux backend plugin loading)
    if system == "Linux":
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if str(lib_dir) not in ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld_path}" if ld_path else str(lib_dir)
            logger.info(f"Set LD_LIBRARY_PATH to include: {lib_dir}")

    # FIRST: Try calling from the CFFI-loaded libllama library
    # This is CRITICAL - we must use the same library context as model loading
    # On macOS, ggml_backend_load_all is typically linked into libllama.dylib
    # On Linux b7376+, it's also available in libllama.so (re-exported from libggml)
    lib = get_lib()

    # Try ggml_backend_load_all_from_path first - this explicitly tells llama.cpp
    # where to find backend plugins, which is more reliable than the default search
    lib_dir_str = str(lib_dir)
    try:
        logger.debug(f"Calling ggml_backend_load_all_from_path({lib_dir_str}) from libllama (CFFI)...")
        lib.ggml_backend_load_all_from_path(lib_dir_str.encode('utf-8'))
        logger.info(f"GGML backends loaded from path {lib_dir_str} via libllama (CFFI)")
        return
    except (AttributeError, OSError) as e:
        logger.debug(f"ggml_backend_load_all_from_path not available in libllama: {e}")
    except Exception as e:
        logger.debug(f"Error calling ggml_backend_load_all_from_path from libllama: {e}")

    # Fallback to ggml_backend_load_all() without explicit path
    try:
        logger.debug("Calling ggml_backend_load_all() from libllama (CFFI)...")
        lib.ggml_backend_load_all()
        logger.info("GGML backends loaded from libllama (CFFI)")
        return
    except (AttributeError, OSError) as e:
        logger.debug(f"ggml_backend_load_all not available in libllama: {e}")
    except Exception as e:
        # CFFI ffi.error or other issues
        logger.debug(f"Error calling ggml_backend_load_all from libllama: {e}")

    # FALLBACK: Try loading from ggml library via ctypes
    # On Windows, ggml_backend_load_all is only in ggml.dll, not llama.dll
    # On Linux, this is less reliable because it's a different library context
    ggml_lib_path = None
    if system == "Windows":
        ggml_lib_path = lib_dir / "ggml.dll"
    elif system == "Linux":
        # Try versioned first, then unversioned
        for pattern in ["libggml.so.0", "libggml.so"]:
            candidate = lib_dir / pattern
            if candidate.exists():
                ggml_lib_path = candidate
                logger.info(f"Found ggml library: {candidate}")
                break
        if ggml_lib_path is None:
            candidates = list(lib_dir.glob("libggml.so*"))
            if candidates:
                ggml_lib_path = candidates[0]
                logger.info(f"Found ggml library via glob: {ggml_lib_path}")

    if ggml_lib_path and ggml_lib_path.exists():
        try:
            import ctypes
            logger.debug(f"FALLBACK: Loading ggml library from {ggml_lib_path}...")

            # Load with RTLD_GLOBAL on Linux so backend registrations are visible globally
            # On Windows, the mode parameter is ignored by ctypes
            if system == "Linux":
                RTLD_GLOBAL = 0x100
                RTLD_NOW = 0x2
                ggml_lib = ctypes.CDLL(str(ggml_lib_path), mode=RTLD_GLOBAL | RTLD_NOW)
            else:
                ggml_lib = ctypes.CDLL(str(ggml_lib_path))

            # CRITICAL for Windows: ggml_backend_load_all() searches for plugins relative
            # to the executable (Python.exe), NOT relative to ggml.dll. We MUST use
            # ggml_backend_load_all_from_path() with the explicit cache directory.
            try:
                logger.debug(f"Calling ggml_backend_load_all_from_path({lib_dir_str}) from {ggml_lib_path.name}...")
                ggml_backend_load_all_from_path = ggml_lib.ggml_backend_load_all_from_path
                ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
                ggml_backend_load_all_from_path.restype = None
                ggml_backend_load_all_from_path(lib_dir_str.encode('utf-8'))

                logger.info(f"GGML backends loaded from path {lib_dir_str} via {ggml_lib_path.name} (fallback)")
                return
            except (AttributeError, OSError) as e:
                logger.debug(f"ggml_backend_load_all_from_path not available in {ggml_lib_path.name}: {e}")

            # Fall back to ggml_backend_load_all() without explicit path
            logger.debug(f"Calling ggml_backend_load_all() from {ggml_lib_path.name}...")
            ggml_backend_load_all = ggml_lib.ggml_backend_load_all
            ggml_backend_load_all.argtypes = []
            ggml_backend_load_all.restype = None
            ggml_backend_load_all()

            logger.info(f"GGML backends loaded from {ggml_lib_path} (fallback)")
            return
        except Exception as e:
            logger.warning(f"Failed to load backends from {ggml_lib_path}: {e}")

    # If we get here, backends couldn't be loaded - this may cause model loading to fail
    logger.warning(
        "Could not load GGML backends. Model loading may fail with "
        "'no backends are loaded' error. "
        f"Searched in: {lib_dir}"
    )


def init_backend():
    """Initialize the llama.cpp backend."""
    logger.debug("Initializing llama.cpp backend...")

    lib = get_lib()

    # Set up logging callback to route llama.cpp logs through Python logging
    _setup_log_callback(lib)

    lib.llama_backend_init()
    logger.info("llama.cpp backend initialized")

    # Load all available compute backends (CPU, CUDA, Metal, Vulkan, etc.)
    # Required for llama.cpp b7376+ - without this, model loading fails with
    # "no backends are loaded" error
    _load_ggml_backends()
    logger.debug("GGML backends loading completed")


def free_backend():
    """Free the llama.cpp backend."""
    lib = get_lib()
    lib.llama_backend_free()
    logger.debug("llama.cpp backend freed")


# Initialize backend on module load
_backend_initialized = False


def ensure_backend():
    """Ensure the backend is initialized."""
    global _backend_initialized
    if not _backend_initialized:
        init_backend()
        _backend_initialized = True


def set_llama_log_level(level: int):
    """Set the log level for llama.cpp output.

    This controls the verbosity of llama.cpp's internal logging.
    Use standard Python logging levels:
        - logging.DEBUG (10): Show all llama.cpp output
        - logging.INFO (20): Show info and above
        - logging.WARNING (30): Show warnings and errors only
        - logging.ERROR (40): Show errors only

    Example:
        import logging
        from llamafarm_llama._bindings import set_llama_log_level
        set_llama_log_level(logging.WARNING)  # Suppress debug/info output
    """
    llama_logger = logging.getLogger("llamafarm_llama.llama_cpp")
    llama_logger.setLevel(level)
