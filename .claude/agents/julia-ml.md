# Julia ML/AI Agent - Best Practices

This agent covers Julia best practices for machine learning, AI APIs, LLM inference, and related tooling.

## Julia Package Manager Basics

Always use Julia's built-in package manager:

```julia
# Enter package mode with ]
# Or use Pkg directly:
using Pkg
Pkg.add("PackageName")
Pkg.add(url="https://github.com/org/Package.jl")  # Unregistered packages
```

## Oxygen.jl - REST API Framework

[Oxygen.jl](https://github.com/OxygenFramework/Oxygen.jl) is Julia's FastAPI-equivalent for building REST APIs.

### Installation
```julia
using Pkg
Pkg.add("Oxygen")
```

### Basic API Setup
```julia
using Oxygen
using HTTP

# Simple GET endpoint
@get "/hello" function()
    return "Hello, World!"
end

# Path parameters
@get "/users/{id}" function(req, id::Int)
    return Dict("user_id" => id)
end

# POST with JSON body
@post "/users" function(req)
    data = json(req)
    return Dict("created" => data)
end

# Query parameters
@get "/search" function(req)
    query = queryparams(req)
    term = get(query, "q", "")
    return Dict("results" => [], "query" => term)
end

# Start server
serve(host="0.0.0.0", port=8080)
```

### Extractors (Reduce Boilerplate)
```julia
using Oxygen

# Define request struct
struct CreateUser
    name::String
    email::String
end

# Oxygen auto-deserializes JSON to struct
@post "/users" function(req, user::Json{CreateUser})
    return Dict("created" => user.payload)
end
```

### Auto-Generated Swagger Docs
Oxygen automatically generates OpenAPI docs at `/docs`:
```julia
serve()  # Visit http://localhost:8080/docs
```

### Parallel Processing
```julia
# Use serveparallel() for multi-threaded handling
serveparallel(host="0.0.0.0", port=8080)
```

### Best Practices
- Use `@get`, `@post`, `@put`, `@patch`, `@delete` macros for routing
- Define structs for request/response bodies
- Use extractors (`Json{}`, `Path{}`, `Query{}`) for automatic parsing
- Enable CORS with `configdocs(cors=true)` for frontend integration
- Use `serveparallel()` for production workloads

---

## LLM Inference with Julia

### Llama2.jl - Native Julia LLM Inference

[Llama2.jl](https://github.com/cafaxo/Llama2.jl) provides native Julia support for Llama-style models with quantization from llama.cpp.

```julia
# Install (not registered, use URL)
using Pkg
Pkg.add(url="https://github.com/cafaxo/Llama2.jl")

using Llama2

# Load model
model = load_model("path/to/model.bin")

# Generate text
output = generate(model, "Once upon a time", max_tokens=100)
println(output)
```

### Calling llama.cpp Server from Julia
If running llama.cpp's server externally:

```julia
using HTTP
using JSON3

function llm_complete(prompt::String;
                      url="http://localhost:8080/completion",
                      max_tokens=256)
    response = HTTP.post(url,
        ["Content-Type" => "application/json"],
        JSON3.write(Dict(
            "prompt" => prompt,
            "n_predict" => max_tokens,
            "temperature" => 0.7
        ))
    )
    return JSON3.read(response.body)["content"]
end

# Usage
result = llm_complete("Explain quantum computing in simple terms:")
```

### Calling OpenAI-Compatible APIs
```julia
using HTTP
using JSON3

function chat_completion(messages::Vector;
                         model="gpt-4",
                         api_key=ENV["OPENAI_API_KEY"],
                         base_url="https://api.openai.com/v1")
    response = HTTP.post("$base_url/chat/completions",
        ["Content-Type" => "application/json",
         "Authorization" => "Bearer $api_key"],
        JSON3.write(Dict(
            "model" => model,
            "messages" => messages
        ))
    )
    return JSON3.read(response.body)["choices"][1]["message"]["content"]
end

# Usage
result = chat_completion([
    Dict("role" => "user", "content" => "Hello!")
])
```

---

## Transformers.jl - NLP & Transformer Models

[Transformers.jl](https://github.com/chengchingwen/Transformers.jl) provides Julia-native transformer implementations with HuggingFace integration.

### Installation
```julia
using Pkg
Pkg.add("Transformers")
Pkg.add("CUDA")  # For GPU support
```

### Loading HuggingFace Models
```julia
using Transformers
using Transformers.HuggingFace

# Load tokenizer and model
tokenizer = hgf"bert-base-uncased:tokenizer"
model = hgf"bert-base-uncased:model"

# Or explicitly
tokenizer = HuggingFace.load_tokenizer("bert-base-uncased")
model = HuggingFace.load_model("bert-base-uncased")
```

### Text Classification
```julia
using Transformers
using Flux

# Tokenize input
text = "This movie was fantastic!"
tokens = tokenizer(text)

# Get embeddings
embeddings = model(tokens)

# Add classification head
classifier = Chain(
    Dense(768, 256, relu),
    Dense(256, 2),
    softmax
)

# Forward pass
logits = classifier(embeddings[:, 1, :])  # Use [CLS] token
```

### Sequence-to-Sequence (Translation, Summarization)
```julia
using Transformers

# Load T5 or similar
tokenizer = hgf"t5-small:tokenizer"
model = hgf"t5-small:model"

# Generate
input_ids = tokenizer("translate English to French: Hello, how are you?")
output = model.generate(input_ids, max_length=50)
result = tokenizer.decode(output)
```

### Best Practices
- Use CUDA.jl for GPU acceleration
- Batch inputs for efficiency
- Cache tokenized datasets
- Use `@threads` for parallel tokenization
- Fine-tune with Flux.jl optimizers

---

## MLJ.jl - Machine Learning Framework

[MLJ.jl](https://github.com/JuliaAI/MLJ.jl) is Julia's scikit-learn equivalent with 200+ models.

### Installation
```julia
using Pkg
Pkg.add("MLJ")
Pkg.add("MLJDecisionTreeInterface")  # For tree models
Pkg.add("MLJLinearModels")           # For linear classifiers
```

### Classification Example
```julia
using MLJ

# Load data
X, y = @load_iris

# Split data
train, test = partition(eachindex(y), 0.8, shuffle=true)

# Load model
Tree = @load DecisionTreeClassifier pkg=DecisionTree

# Create and train
tree = Tree(max_depth=5)
mach = machine(tree, X, y)
fit!(mach, rows=train)

# Predict
ŷ = predict(mach, rows=test)
accuracy = mean(mode.(ŷ) .== y[test])
```

### Available Classifiers
```julia
# List all classifiers
models(m -> m.is_supervised && m.prediction_type == :probabilistic)

# Common classifiers:
# - DecisionTreeClassifier (DecisionTree.jl)
# - RandomForestClassifier (DecisionTree.jl)
# - LogisticClassifier (MLJLinearModels.jl)
# - KNNClassifier (NearestNeighborModels.jl)
# - SVMClassifier (LIBSVM.jl)
# - XGBoostClassifier (XGBoost.jl)
```

### Hyperparameter Tuning
```julia
using MLJ

Tree = @load DecisionTreeClassifier pkg=DecisionTree

# Define search range
r = range(Tree(), :max_depth, lower=1, upper=20)

# Grid search
tuned_tree = TunedModel(
    model=Tree(),
    tuning=Grid(resolution=10),
    range=r,
    measure=accuracy
)

mach = machine(tuned_tree, X, y)
fit!(mach)

# Best model
best = fitted_params(mach).best_model
```

### Pipelines
```julia
using MLJ

# Create pipeline
pipe = @pipeline(
    Standardizer(),
    PCA(maxoutdim=10),
    DecisionTreeClassifier()
)

mach = machine(pipe, X, y)
fit!(mach)
```

---

## OutlierDetection.jl - Anomaly Detection

[OutlierDetection.jl](https://github.com/OutlierDetectionJL/OutlierDetection.jl) provides anomaly detection built on MLJ.

### Installation
```julia
using Pkg
Pkg.add("OutlierDetection")
```

### Basic Anomaly Detection
```julia
using OutlierDetection
using MLJ

# Load detector
KNN = @load KNNDetector pkg=OutlierDetectionNeighbors

# Create detector (k=5 neighbors)
detector = KNN(k=5)

# Fit and score
mach = machine(detector, X)
fit!(mach)
scores = transform(mach, X)

# Higher score = more anomalous
threshold = quantile(scores, 0.95)
anomalies = scores .> threshold
```

### Available Detectors
```julia
# Neighbor-based
KNNDetector      # K-nearest neighbors distance
LOFDetector      # Local Outlier Factor
DNNDetector      # Distance to neighbors in hypersphere

# Statistical
ABODDetector     # Angle-based outlier detection

# Neural Network (OutlierDetectionNetworks.jl)
AutoEncoderDetector
```

### Time Series Anomaly Detection
```julia
using OutlierDetection
using OutlierDetectionData

# Load time series dataset
X, y = load_dataset("TSAD", "ECG")

# Use sliding window approach
window_size = 50
X_windowed = sliding_window(X, window_size)

# Detect anomalies
detector = KNNDetector(k=10)
mach = machine(detector, X_windowed)
fit!(mach)
scores = transform(mach, X_windowed)
```

### Best Practices
- Normalize data before detection
- Use multiple detectors and ensemble scores
- Tune k/threshold on validation set with known anomalies
- For time series: use sliding windows or specialized temporal methods

---

## Embeddings.jl - Word & Text Embeddings

[Embeddings.jl](https://github.com/JuliaText/Embeddings.jl) provides access to pretrained embeddings (Word2Vec, GloVe, FastText).

### Installation
```julia
using Pkg
Pkg.add("Embeddings")
```

### Loading Pretrained Embeddings
```julia
using Embeddings

# Load GloVe (downloads automatically)
embtable = load_embeddings(GloVe)

# Get embedding for a word
word = "computer"
idx = findfirst(==(word), embtable.vocab)
embedding = embtable.embeddings[:, idx]

# Create lookup function
function get_embedding(word)
    idx = findfirst(==(word), embtable.vocab)
    isnothing(idx) ? zeros(size(embtable.embeddings, 1)) : embtable.embeddings[:, idx]
end
```

### Available Embeddings
```julia
# Word2Vec (Google News, 300d)
load_embeddings(Word2Vec)

# GloVe (various sizes)
load_embeddings(GloVe{:en}, 1)  # 50d
load_embeddings(GloVe{:en}, 2)  # 100d
load_embeddings(GloVe{:en}, 3)  # 200d
load_embeddings(GloVe{:en}, 4)  # 300d

# FastText (many languages)
load_embeddings(FastText{:en})  # English
load_embeddings(FastText{:es})  # Spanish
load_embeddings(FastText{:fr})  # French
```

### Sentence Embeddings (Simple Average)
```julia
function sentence_embedding(sentence::String, embtable)
    words = split(lowercase(sentence))
    embeddings = [get_embedding(w, embtable) for w in words]
    return mean(embeddings)
end

# Usage
sent_emb = sentence_embedding("The quick brown fox", embtable)
```

### Semantic Similarity
```julia
using LinearAlgebra

function cosine_similarity(a, b)
    return dot(a, b) / (norm(a) * norm(b))
end

# Compare sentences
emb1 = sentence_embedding("I love programming", embtable)
emb2 = sentence_embedding("Coding is my passion", embtable)
similarity = cosine_similarity(emb1, emb2)
```

---

## Building an ML API with Oxygen

Complete example combining Oxygen + MLJ + Embeddings:

```julia
using Oxygen
using MLJ
using Embeddings
using JSON3

# Load models at startup
const EMBEDDINGS = load_embeddings(GloVe)
const CLASSIFIER = begin
    # Load your trained model
    mach = machine("model.jlso")
    mach
end

# Health check
@get "/health" function()
    return Dict("status" => "healthy")
end

# Classify text
@post "/classify" function(req)
    data = json(req)
    text = data["text"]

    # Get embedding
    emb = sentence_embedding(text, EMBEDDINGS)

    # Classify
    pred = predict(CLASSIFIER, reshape(emb, 1, :))

    return Dict(
        "text" => text,
        "prediction" => mode(pred[1]),
        "probabilities" => pdf.(pred, levels(pred[1]))
    )
end

# Get embeddings
@post "/embed" function(req)
    data = json(req)
    text = data["text"]
    emb = sentence_embedding(text, EMBEDDINGS)
    return Dict("embedding" => collect(emb))
end

# Anomaly detection
@post "/anomaly" function(req)
    data = json(req)
    features = Float64.(data["features"])

    # Score with detector
    score = transform(ANOMALY_DETECTOR, reshape(features, 1, :))[1]

    return Dict(
        "score" => score,
        "is_anomaly" => score > THRESHOLD
    )
end

serve(host="0.0.0.0", port=8080)
```

---

## Project Structure Best Practices

```
my_julia_ml_project/
├── Project.toml          # Dependencies
├── Manifest.toml         # Lock file
├── src/
│   ├── MyProject.jl      # Main module
│   ├── models/
│   │   ├── classifier.jl
│   │   └── anomaly.jl
│   ├── api/
│   │   ├── routes.jl
│   │   └── handlers.jl
│   └── utils/
│       ├── embeddings.jl
│       └── preprocessing.jl
├── test/
│   └── runtests.jl
├── scripts/
│   ├── train.jl
│   └── serve.jl
└── models/               # Saved model files
    └── classifier.jlso
```

### Project.toml Example
```toml
name = "MyMLProject"
uuid = "..."
version = "0.1.0"

[deps]
Oxygen = "df9a0d86-3283-4920-82dc-4555fc0d1d8b"
MLJ = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
Transformers = "21ca0261-441d-5938-ace7-c90938fde4d4"
OutlierDetection = "..."
Embeddings = "..."
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
```

---

## Performance Tips

1. **Type Stability**: Always use concrete types in hot paths
2. **Pre-allocate**: Use `similar()` or pre-allocate arrays
3. **GPU**: Use CUDA.jl for matrix operations
4. **Threading**: Use `Threads.@threads` for parallel loops
5. **Profiling**: Use `@time`, `@btime`, `Profile.jl`

```julia
# Good: Type-stable
function process(x::Vector{Float64})::Vector{Float64}
    return x .^ 2
end

# Good: Pre-allocate
function batch_embed!(result, texts, embtable)
    @threads for i in eachindex(texts)
        result[:, i] .= sentence_embedding(texts[i], embtable)
    end
end
```

---

## Documentation Links

- [Oxygen.jl Docs](https://oxygenframework.github.io/Oxygen.jl/stable/)
- [MLJ.jl Docs](https://juliaai.github.io/MLJ.jl/stable/)
- [Transformers.jl Docs](https://chengchingwen.github.io/Transformers.jl/dev/)
- [OutlierDetection.jl Docs](https://outlierdetectionjl.github.io/OutlierDetection.jl/dev/)
- [Embeddings.jl GitHub](https://github.com/JuliaText/Embeddings.jl)
- [Llama2.jl GitHub](https://github.com/cafaxo/Llama2.jl)
- [Flux.jl (Deep Learning)](https://fluxml.ai/)
- [JuliaHub Blog: LLM Tutorial](https://juliahub.com/blog/large-language-model-llm-tutorial-with-julias-transformers-jl)
