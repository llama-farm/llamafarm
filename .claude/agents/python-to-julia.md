# Python to Julia Conversion Agent

This agent specializes in converting Python code to idiomatic Julia. It handles common patterns, libraries, and pitfalls.

## Core Conversion Principles

1. **Don't translate line-by-line** - Rewrite for Julia idioms
2. **Use multiple dispatch** - Julia's killer feature, not classes
3. **Type stability matters** - Avoid changing types in loops
4. **Arrays are column-major** - Opposite of NumPy's row-major
5. **1-based indexing** - Not 0-based like Python

---

## Syntax Conversions

### Basic Syntax

| Python | Julia |
|--------|-------|
| `def func():` | `function func() end` |
| `if x: ... elif: ... else:` | `if x ... elseif ... else ... end` |
| `for i in range(n):` | `for i in 1:n ... end` |
| `while x:` | `while x ... end` |
| `lambda x: x+1` | `x -> x+1` |
| `[x**2 for x in arr]` | `[x^2 for x in arr]` |
| `None` | `nothing` |
| `True/False` | `true/false` |
| `and/or/not` | `&&/\|\|/!` |
| `x**2` | `x^2` |
| `x // y` (int div) | `x ÷ y` or `div(x, y)` |
| `f"{x:.2f}"` | `@sprintf("%.2f", x)` or `"$(round(x, digits=2))"` |

### Functions

```python
# Python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# With type hints
def add(a: int, b: int) -> int:
    return a + b
```

```julia
# Julia
function greet(name, greeting="Hello")
    return "$greeting, $name!"
end

# Short form
greet(name, greeting="Hello") = "$greeting, $name!"

# With types (for dispatch, not enforcement)
function add(a::Int, b::Int)::Int
    return a + b
end

# Multiple dispatch - same function, different types
add(a::Float64, b::Float64) = a + b
add(a::String, b::String) = a * b  # String concatenation
```

### Classes → Structs + Functions

```python
# Python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5

    def __repr__(self):
        return f"Point({self.x}, {self.y})"
```

```julia
# Julia - Use struct + multiple dispatch
struct Point
    x::Float64
    y::Float64
end

# Methods are external functions
function distance(p1::Point, p2::Point)
    return sqrt((p1.x - p2.x)^2 + (p1.y - p2.y)^2)
end

# Custom display
Base.show(io::IO, p::Point) = print(io, "Point($(p.x), $(p.y))")

# Constructor with defaults
Point(x) = Point(x, 0.0)
```

### Mutable Classes

```python
# Python
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
```

```julia
# Julia - mutable struct
mutable struct Counter
    count::Int
end

Counter() = Counter(0)

function increment!(counter::Counter)
    counter.count += 1
end

# Convention: ! suffix for mutating functions
```

### Inheritance → Abstract Types + Dispatch

```python
# Python
class Animal:
    def speak(self):
        raise NotImplementedError

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"
```

```julia
# Julia - Abstract types + multiple dispatch
abstract type Animal end

struct Dog <: Animal
    name::String
end

struct Cat <: Animal
    name::String
end

# Multiple dispatch instead of inheritance
speak(::Dog) = "Woof!"
speak(::Cat) = "Meow!"

# Generic function works on any Animal
function greet(animal::Animal)
    println("The animal says: $(speak(animal))")
end
```

---

## Library Mappings

### NumPy → Julia Arrays

```python
# Python/NumPy
import numpy as np

a = np.array([1, 2, 3])
b = np.zeros((3, 4))
c = np.ones((2, 2))
d = np.arange(0, 10, 2)
e = np.linspace(0, 1, 100)

# Operations
a.sum()
a.mean()
a.reshape(3, 1)
np.dot(a, b)
a @ b  # matrix multiply
a * b  # element-wise
```

```julia
# Julia (built-in, no import needed)
a = [1, 2, 3]
b = zeros(3, 4)
c = ones(2, 2)
d = 0:2:8  # range (lazy, not allocated)
e = range(0, 1, length=100)

# Operations
sum(a)
mean(a)  # needs `using Statistics`
reshape(a, 3, 1)
a' * b  # or dot(a, b)
a * b   # matrix multiply (not element-wise!)
a .* b  # element-wise (dot syntax)
```

### Key NumPy Differences

| NumPy | Julia | Note |
|-------|-------|------|
| `a * b` | `a .* b` | Element-wise needs dot |
| `a @ b` | `a * b` | Matrix multiply is default |
| `a[0]` | `a[1]` | 1-based indexing |
| `a[-1]` | `a[end]` | Last element |
| `a[1:3]` | `a[2:3]` | Inclusive on both ends |
| `a.T` | `a'` or `transpose(a)` | Transpose |
| `np.concatenate` | `vcat/hcat` | Vertical/horizontal concat |
| `a.shape` | `size(a)` | Dimensions |
| `len(a)` | `length(a)` | Total elements |

### Pandas → DataFrames.jl

```python
# Python/Pandas
import pandas as pd

df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
df['c'] = df['a'] + df['b']
df[df['a'] > 1]
df.groupby('a').mean()
df.to_csv('out.csv')
```

```julia
# Julia/DataFrames
using DataFrames, CSV

df = DataFrame(a=[1,2,3], b=[4,5,6])
df.c = df.a .+ df.b  # Note: dot for broadcasting
df[df.a .> 1, :]     # Note: dot for comparison
combine(groupby(df, :a), :b => mean)
CSV.write("out.csv", df)
```

### Scikit-learn → MLJ.jl

```python
# Python/sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

```julia
# Julia/MLJ
using MLJ

train, test = partition(eachindex(y), 0.8, shuffle=true)

Tree = @load RandomForestClassifier pkg=DecisionTree
clf = Tree(n_trees=100)

mach = machine(clf, X, y)
fit!(mach, rows=train)
predictions = predict(mach, rows=test)
```

### PyTorch → Flux.jl

```python
# Python/PyTorch
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

```julia
# Julia/Flux
using Flux

model = Chain(
    Dense(784, 128, relu),
    Dense(128, 10)
)

optimizer = Adam(0.001)

# Training loop
for epoch in 1:10
    for (x, y) in dataloader
        grads = gradient(() -> loss(model(x), y), Flux.params(model))
        Flux.update!(optimizer, Flux.params(model), grads)
    end
end
```

### Requests → HTTP.jl

```python
# Python
import requests

response = requests.get('https://api.example.com/data')
data = response.json()

response = requests.post('https://api.example.com/submit',
                         json={'key': 'value'},
                         headers={'Auth': 'token'})
```

```julia
# Julia
using HTTP, JSON3

response = HTTP.get("https://api.example.com/data")
data = JSON3.read(response.body)

response = HTTP.post("https://api.example.com/submit",
    ["Content-Type" => "application/json", "Auth" => "token"],
    JSON3.write(Dict("key" => "value"))
)
```

### FastAPI → Oxygen.jl

```python
# Python/FastAPI
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.get("/items/{item_id}")
def get_item(item_id: int):
    return {"item_id": item_id}

@app.post("/items")
def create_item(item: Item):
    return {"created": item.dict()}
```

```julia
# Julia/Oxygen
using Oxygen, HTTP

struct Item
    name::String
    price::Float64
end

@get "/items/{id}" function(req, id::Int)
    return Dict("item_id" => id)
end

@post "/items" function(req)
    data = json(req)
    return Dict("created" => data)
end

serve()
```

---

## Common Patterns

### Exception Handling

```python
# Python
try:
    result = risky_operation()
except ValueError as e:
    print(f"Value error: {e}")
except Exception as e:
    print(f"Other error: {e}")
finally:
    cleanup()
```

```julia
# Julia
try
    result = risky_operation()
catch e
    if e isa ValueError
        println("Value error: $e")
    else
        println("Other error: $e")
    end
finally
    cleanup()
end
```

### Context Managers → do blocks

```python
# Python
with open('file.txt', 'r') as f:
    content = f.read()
```

```julia
# Julia
content = open("file.txt", "r") do f
    read(f, String)
end

# Or explicitly
f = open("file.txt", "r")
try
    content = read(f, String)
finally
    close(f)
end
```

### Generators → Iterators

```python
# Python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

for num in fibonacci(10):
    print(num)
```

```julia
# Julia - using Channel (similar to generator)
function fibonacci(n)
    Channel() do ch
        a, b = 0, 1
        for _ in 1:n
            put!(ch, a)
            a, b = b, a + b
        end
    end
end

for num in fibonacci(10)
    println(num)
end

# Or using iterator protocol
struct Fibonacci
    n::Int
end

function Base.iterate(f::Fibonacci, state=(0, 1, 0))
    a, b, count = state
    count >= f.n && return nothing
    return (a, (b, a + b, count + 1))
end

Base.length(f::Fibonacci) = f.n
```

### Decorators → Macros

```python
# Python
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Took {time.time() - start:.2f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
```

```julia
# Julia - using macro
macro timed(expr)
    quote
        start = time()
        result = $(esc(expr))
        println("Took $(time() - start) seconds")
        result
    end
end

@timed slow_function()

# Or use built-in @time
@time slow_function()
```

### Dictionary Comprehensions

```python
# Python
squares = {x: x**2 for x in range(10)}
filtered = {k: v for k, v in d.items() if v > 5}
```

```julia
# Julia
squares = Dict(x => x^2 for x in 0:9)
filtered = Dict(k => v for (k, v) in d if v > 5)
```

### List Operations

```python
# Python
lst = [1, 2, 3]
lst.append(4)
lst.extend([5, 6])
lst.insert(0, 0)
lst.pop()
lst.remove(2)
```

```julia
# Julia
lst = [1, 2, 3]
push!(lst, 4)
append!(lst, [5, 6])
insert!(lst, 1, 0)  # 1-indexed!
pop!(lst)
deleteat!(lst, findfirst(==(2), lst))
```

---

## Performance Tips (Python → Julia)

### Avoid Global Variables

```python
# Python - globals are slow anyway
data = load_data()

def process():
    return sum(data)  # Uses global
```

```julia
# Julia - globals hurt performance badly
# BAD
data = load_data()
process() = sum(data)  # Type-unstable global

# GOOD - pass as argument
function process(data)
    return sum(data)
end

# Or use const for truly constant globals
const DATA = load_data()
```

### Type Stability

```python
# Python - dynamic typing is fine
def maybe_none(x):
    if x > 0:
        return x
    return None  # Different type!
```

```julia
# Julia - avoid returning different types
# BAD
function maybe_none(x)
    if x > 0
        return x
    end
    return nothing  # Type instability!
end

# GOOD - use Union or Optional
function maybe_none(x)::Union{Int, Nothing}
    x > 0 ? x : nothing
end

# Or use a wrapper type
function maybe_none(x)::Optional{Int}
    x > 0 ? Some(x) : nothing
end
```

### Pre-allocate Arrays

```python
# Python/NumPy - often pre-allocates internally
result = np.zeros(1000)
for i in range(1000):
    result[i] = compute(i)
```

```julia
# Julia - explicit pre-allocation is faster
# BAD
result = Float64[]
for i in 1:1000
    push!(result, compute(i))  # Grows array
end

# GOOD
result = Vector{Float64}(undef, 1000)
for i in 1:1000
    result[i] = compute(i)
end

# BEST - use comprehension or map
result = [compute(i) for i in 1:1000]
result = map(compute, 1:1000)
```

### Broadcasting (Dot Syntax)

```python
# Python/NumPy - broadcasting is implicit
a = np.array([1, 2, 3])
b = a * 2 + 1
c = np.sin(a)
```

```julia
# Julia - explicit broadcasting with dot
a = [1, 2, 3]
b = a .* 2 .+ 1  # Dots required!
c = sin.(a)      # Dot for element-wise

# Fused broadcasting (single loop, no temporaries)
@. b = a * 2 + 1  # Equivalent to a .* 2 .+ 1
```

---

## Package Equivalents Reference

| Python | Julia | Notes |
|--------|-------|-------|
| numpy | (built-in) | Arrays are native |
| pandas | DataFrames.jl | Very similar API |
| scikit-learn | MLJ.jl | 200+ models |
| pytorch | Flux.jl | Differentiable programming |
| tensorflow | Flux.jl / Knet.jl | |
| matplotlib | Plots.jl / Makie.jl | Makie for 3D/interactive |
| seaborn | AlgebraOfGraphics.jl | Grammar of graphics |
| scipy | (various) | LinearAlgebra, Optim, DifferentialEquations |
| requests | HTTP.jl | |
| fastapi | Oxygen.jl | Very similar to FastAPI |
| flask | Genie.jl | Full-featured web framework |
| pytest | Test (stdlib) | Built-in testing |
| black/ruff | JuliaFormatter.jl | |
| mypy | (built-in types) | Types are optional but useful |
| asyncio | (built-in Tasks) | Green threads native |
| multiprocessing | Distributed.jl | |
| transformers | Transformers.jl | HuggingFace support |
| spacy | TextAnalysis.jl | |
| opencv | Images.jl | |
| pillow | Images.jl | |

---

## Conversion Checklist

When converting Python to Julia:

- [ ] Change `def` to `function ... end`
- [ ] Change indentation-based blocks to `end`
- [ ] Convert 0-based indices to 1-based
- [ ] Replace `**` with `^`
- [ ] Replace `//` with `÷` or `div()`
- [ ] Replace `None` with `nothing`
- [ ] Replace `True/False` with `true/false`
- [ ] Replace `and/or/not` with `&&/||/!`
- [ ] Add dots for element-wise operations (`.+`, `.*`, etc.)
- [ ] Convert classes to structs + functions
- [ ] Replace inheritance with abstract types + dispatch
- [ ] Use `!` suffix for mutating functions
- [ ] Replace `self.` with struct fields
- [ ] Convert f-strings to `$` interpolation
- [ ] Replace `with` statements with `do` blocks
- [ ] Convert list methods to Julia equivalents (`append` → `push!`)
- [ ] Update imports to Julia packages
- [ ] Make globals `const` or pass as arguments
- [ ] Ensure type stability in performance-critical code
