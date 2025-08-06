# Python Programming Fundamentals

## Introduction to Python

Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum in 1991, Python has become one of the most popular programming languages in the world.

### Key Features
- **Easy to learn and read**: Clean, intuitive syntax
- **Cross-platform**: Runs on Windows, Mac, Linux, and more
- **Interpreted**: No compilation step required
- **Dynamically typed**: Variable types determined at runtime
- **Object-oriented**: Supports OOP principles
- **Extensive libraries**: Vast ecosystem of third-party packages

## Basic Syntax and Data Types

### Variables and Assignment
```python
# Variable assignment
name = "Alice"
age = 30
height = 5.6
is_student = True

# Multiple assignment
x, y, z = 1, 2, 3
```

### Data Types
- **Numbers**: `int`, `float`, `complex`
- **Strings**: `str` (text data)
- **Booleans**: `bool` (True/False)
- **None**: Represents absence of value

### String Operations
```python
# String methods
text = "Hello, World!"
print(text.upper())      # HELLO, WORLD!
print(text.lower())      # hello, world!
print(text.replace("Hello", "Hi"))  # Hi, World!

# String formatting
name = "Bob"
age = 25
message = f"My name is {name} and I am {age} years old"
```

## Control Flow

### Conditional Statements
```python
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B" 
elif score >= 70:
    grade = "C"
else:
    grade = "F"
```

### Loops
```python
# For loop
fruits = ["apple", "banana", "orange"]
for fruit in fruits:
    print(f"I like {fruit}")

# While loop
count = 0
while count < 5:
    print(f"Count: {count}")
    count += 1

# Range function
for i in range(1, 6):  # 1, 2, 3, 4, 5
    print(i)
```

## Data Structures

### Lists
```python
# Creating and modifying lists
numbers = [1, 2, 3, 4, 5]
numbers.append(6)        # Add element
numbers.insert(0, 0)     # Insert at index
numbers.remove(3)        # Remove specific value
popped = numbers.pop()   # Remove and return last element

# List comprehension
squares = [x**2 for x in range(1, 6)]  # [1, 4, 9, 16, 25]
```

### Dictionaries
```python
# Creating dictionaries
person = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}

# Accessing and modifying
print(person["name"])    # Alice
person["job"] = "Engineer"  # Add new key-value pair
person.update({"age": 31})  # Update existing value

# Dictionary methods
keys = person.keys()     # Get all keys
values = person.values() # Get all values
items = person.items()   # Get key-value pairs
```

### Tuples and Sets
```python
# Tuples (immutable)
coordinates = (10, 20)
x, y = coordinates  # Unpacking

# Sets (unique elements)
unique_numbers = {1, 2, 3, 3, 4}  # {1, 2, 3, 4}
unique_numbers.add(5)
unique_numbers.remove(1)
```

## Functions

### Function Definition
```python
def greet(name, greeting="Hello"):
    """Function with default parameter"""
    return f"{greeting}, {name}!"

# Function calls
message1 = greet("Alice")           # "Hello, Alice!"
message2 = greet("Bob", "Hi")       # "Hi, Bob!"
```

### Advanced Function Features
```python
# Variable arguments
def sum_all(*args):
    return sum(args)

result = sum_all(1, 2, 3, 4, 5)  # 15

# Keyword arguments
def create_profile(**kwargs):
    return kwargs

profile = create_profile(name="Alice", age=30, city="NYC")

# Lambda functions
square = lambda x: x**2
numbers = [1, 2, 3, 4, 5]
squared = list(map(square, numbers))  # [1, 4, 9, 16, 25]
```

## Object-Oriented Programming

### Classes and Objects
```python
class Dog:
    species = "Canis familiaris"  # Class variable
    
    def __init__(self, name, breed):
        self.name = name      # Instance variable
        self.breed = breed
    
    def bark(self):
        return f"{self.name} says Woof!"
    
    def info(self):
        return f"{self.name} is a {self.breed}"

# Creating objects
dog1 = Dog("Buddy", "Golden Retriever")
dog2 = Dog("Max", "German Shepherd")

print(dog1.bark())  # Buddy says Woof!
print(dog2.info())  # Max is a German Shepherd
```

### Inheritance
```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        pass

class Cat(Animal):  # Inheritance
    def speak(self):
        return f"{self.name} says Meow!"

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

# Using inherited classes
cat = Cat("Whiskers")
dog = Dog("Buddy")
print(cat.speak())  # Whiskers says Meow!
print(dog.speak())  # Buddy says Woof!
```

## Error Handling

### Try/Except Blocks
```python
def divide_numbers(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        return "Cannot divide by zero!"
    except TypeError:
        return "Invalid input types!"
    finally:
        print("Division operation completed")

# Usage
print(divide_numbers(10, 2))  # 5.0
print(divide_numbers(10, 0))  # Cannot divide by zero!
```

## File Operations

### Reading and Writing Files
```python
# Writing to a file
with open("data.txt", "w") as file:
    file.write("Hello, World!\n")
    file.write("Python is awesome!")

# Reading from a file
with open("data.txt", "r") as file:
    content = file.read()
    print(content)

# Reading line by line
with open("data.txt", "r") as file:
    for line in file:
        print(line.strip())
```

## Common Libraries

### Standard Library Modules
```python
import datetime
import random
import os
import json

# Working with dates
today = datetime.date.today()
now = datetime.datetime.now()

# Random numbers
random_number = random.randint(1, 100)
random_choice = random.choice(["apple", "banana", "orange"])

# File system operations
current_dir = os.getcwd()
file_exists = os.path.exists("data.txt")

# JSON operations
data = {"name": "Alice", "age": 30}
json_string = json.dumps(data)
parsed_data = json.loads(json_string)
```

### Popular Third-Party Libraries
- **NumPy**: Numerical computing with arrays
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Requests**: HTTP library for API calls
- **Flask/Django**: Web frameworks
- **TensorFlow/PyTorch**: Machine learning

## Best Practices

### Code Style and PEP 8
- Use 4 spaces for indentation
- Keep lines under 79 characters
- Use lowercase with underscores for variables and functions
- Use CamelCase for class names
- Add docstrings to functions and classes

### Pythonic Code Patterns
```python
# List comprehensions instead of loops
squares = [x**2 for x in range(10)]

# Enumerate for index and value
for index, value in enumerate(["a", "b", "c"]):
    print(f"{index}: {value}")

# Use zip for parallel iteration
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# Context managers for resource management
with open("file.txt", "r") as f:
    content = f.read()
```

## Getting Started

### Installation
1. Download Python from python.org
2. Install using package manager (brew, apt, etc.)
3. Use virtual environments for projects

### Development Environment
- **Text Editors**: VS Code, Sublime Text, Atom
- **IDEs**: PyCharm, Spyder, IDLE
- **Interactive**: Jupyter Notebooks, IPython

### Next Steps
1. Practice with coding challenges
2. Build small projects
3. Explore data science libraries
4. Learn web development frameworks
5. Contribute to open-source projects

Python's versatility makes it suitable for web development, data analysis, machine learning, automation, and more. Its readable syntax and extensive ecosystem make it an excellent choice for both beginners and experienced programmers.