---
layout: post
title: "Python Object-Oriented Programming"
date: 2026-02-15 12:00
categories: [Python, OOP]
tags: [Python, OOP, Classes, Inheritance, Polymorphism, Encapsulation]
---

# Python Object-Oriented Programming: A Beginner's Guide

## Introduction

Object-Oriented Programming, commonly known as OOP, is one of the most important paradigms in modern software development. Unlike procedural programming, where you focus on writing functions that perform operations on data, OOP allows you to think about your program as a collection of interacting objects, each with its own data and behavior. Python, with its clean and readable syntax, makes learning OOP concepts remarkably accessible for beginners. Whether you're building a simple command-line tool or a complex web application, understanding OOP will transform how you write and organize your code.

By the end of this guide, you'll understand how to:

* Design and create your own classes and objects
* Use encapsulation to protect your data
* Implement inheritance to reuse and extend code
* Apply polymorphism to make your programs flexible
* Work with advanced OOP concepts like decorators and generators

---

## Understanding Object-Oriented Programming

### What is OOP?

Imagine you're building a car racing game. In procedural programming, you might have separate variables for each car's speed, position, color, and separate functions to move cars, change their colors, and check collisions. As your game grows, managing all these scattered variables and functions becomes a nightmare. **Object-Oriented Programming** solves this problem by bundling related data and behaviors together into a single entity called an **object**. In our racing game, each car would be an object containing its own speed, position, and color, along with methods to move, accelerate, and change appearance.

OOP is built on four fundamental principles that work together to create organized, maintainable, and scalable code:

* **Encapsulation**: Bundling data and the methods that operate on that data within a single unit (class), hiding internal details from outside interference.

* **Abstraction**: Hiding complex implementation details and showing only the essential features of an object, making it easier to use.

* **Inheritance**: Creating new classes based on existing ones, inheriting their attributes and methods, which promotes code reuse.

* **Polymorphism**: The ability of different objects to respond to the same message or method call in different ways, providing flexibility in your programs.

---

## Classes and Objects

### What is a Class?

A **class** is like a blueprint or template for creating objects. Think of it as a cookie cutter that defines the shape and characteristics of cookies (objects) you'll create. The class defines what attributes (data) and methods (behaviors) every object created from it will have. For example, a `Dog` class might define that all dogs have a name, age, and breed (attributes), and can bark, eat, and sleep (methods). The class itself doesn't represent any specific dog; it's just the definition of what it means to be a dog in your program.

### What is an Object?

An **object** is a concrete instance of a class. Using our cookie cutter analogy, if the class is the cutter, objects are the actual cookies you produce. Each object has its own copy of the attributes defined in the class, but they can have different values. So while all Dog objects have a name attribute, one dog might be named "Buddy" while another is named "Max". Objects are the actual entities that exist in your program's memory and interact with each other.

```python
# Creating a simple class
class Dog:
    """A simple class to represent a dog."""
    
    # Class attribute (shared by all instances)
    species = "Canis familiaris"
    
    # Constructor method - called when creating a new object
    def __init__(self, name, age):
        # Instance attributes (unique to each object)
        self.name = name
        self.age = age
    
    # Instance method - defines behavior
    def bark(self):
        return f"{self.name} says Woof!"
    
    def describe(self):
        return f"{self.name} is {self.age} years old."


# Creating objects (instances) from the Dog class
dog1 = Dog("Buddy", 3)
dog2 = Dog("Max", 5)

# Accessing attributes
print(dog1.name)        # Output: Buddy
print(dog2.age)         # Output: 5

# Calling methods
print(dog1.bark())      # Output: Buddy says Woof!
print(dog2.describe())  # Output: Max is 5 years old.
```

### Understanding `self`

The `self` parameter in Python methods often confuses beginners, but it's actually quite straightforward. **`self`** refers to the specific object that is calling the method. When you write `dog1.bark()`, Python automatically passes `dog1` as the `self` argument to the `bark` method. This is how the method knows which dog's name to use. Without `self`, methods wouldn't know which object's data to work with. You can think of `self` as meaning "this specific object."

---

## Variables in Classes

### Instance Variables

**Instance variables** are attributes that belong to a specific object instance. Each object has its own copy of these variables, and they can have different values for different objects. In the Dog class example above, `name` and `age` are instance variables because each dog has its own name and age. These are typically defined inside the `__init__` method using `self.variable_name`. When you change an instance variable for one object, it doesn't affect other objects of the same class.

```python
class Student:
    def __init__(self, name, student_id):
        # Instance variables - unique for each student
        self.name = name
        self.student_id = student_id
        self.grades = []

# Each student has their own name, ID, and grades
student1 = Student("Alice", "S001")
student2 = Student("Bob", "S002")

student1.grades.append(85)
student2.grades.append(92)

print(student1.grades)  # Output: [85]
print(student2.grades)  # Output: [92] - different from student1
```

### Class Variables

**Class variables** are attributes that are shared by all instances of a class. They belong to the class itself, not to any individual object. This means there's only one copy of a class variable, and when you modify it, the change is visible to all instances. Class variables are defined outside of any method, typically at the top of the class definition. They're useful for storing data that should be the same across all objects, like constants or counters.

```python
class Student:
    # Class variable - shared by all students
    school_name = "UIU"
    total_students = 0
    
    def __init__(self, name):
        self.name = name
        # Incrementing class variable
        Student.total_students += 1

# All students share the same school_name
student1 = Student("Alice")
student2 = Student("Bob")

print(student1.school_name)  # Output: UIU
print(student2.school_name)  # Output: UIU

# Class variable keeps track of total students
print(Student.total_students)  # Output: 2

# Changing class variable affects all instances
Student.school_name = "United International University"
print(student1.school_name)  # Output: United International University
print(student2.school_name)  # Output: United International University
```

---

## Scopes and Namespaces

### Understanding Namespaces

A **namespace** is essentially a mapping from names to objects. Think of it as a dictionary where the keys are variable names and the values are the actual objects. Python has several types of namespaces that exist at different levels. The built-in namespace contains Python's built-in functions like `print()` and `len()`. The global namespace contains names defined at the module level. The local namespace contains names defined within a function or method. When you use a variable name, Python searches these namespaces in a specific order to find the corresponding object.

### Scope Rules

**Scope** defines the region of your code where a particular variable is accessible. Python follows the **LEGB rule** when looking up variable names:

* **L (Local)**: Variables defined inside the current function
* **E (Enclosing)**: Variables in the enclosing function (for nested functions)
* **G (Global)**: Variables defined at the module level
* **B (Built-in)**: Python's built-in names

```python
# Demonstrating scope
x = "global"  # Global scope

def outer_function():
    x = "enclosing"  # Enclosing scope
    
    def inner_function():
        x = "local"  # Local scope
        print(f"Inside inner_function: {x}")
    
    inner_function()
    print(f"Inside outer_function: {x}")

outer_function()
print(f"In global scope: {x}")

# Output:
# Inside inner_function: local
# Inside outer_function: enclosing
# In global scope: global
```

---

## Methods in Python Classes

### Instance Methods

**Instance methods** are the most common type of methods in Python classes. They operate on an instance of the class and have access to the instance through the `self` parameter. Instance methods can read and modify instance variables and call other instance methods. They're used when you need to work with the specific data of an object.

```python
class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    # Instance method - works with instance data
    def area(self):
        import math
        return math.pi * self.radius ** 2
    
    def circumference(self):
        import math
        return 2 * math.pi * self.radius

circle = Circle(5)
print(f"Area: {circle.area():.2f}")           # Output: Area: 78.54
print(f"Circumference: {circle.circumference():.2f}")  # Output: Circumference: 31.42
```

### Class Methods

**Class methods** operate on the class itself rather than on instances. They receive the class as the first argument (conventionally named `cls`) instead of `self`. Class methods are defined using the `@classmethod` decorator. They're useful for creating factory methods or modifying class-level data that affects all instances.

```python
class Student:
    school_name = "UIU"
    
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
    
    # Class method - works with class data
    @classmethod
    def change_school(cls, new_name):
        cls.school_name = new_name
    
    # Factory method - alternative way to create Student objects
    @classmethod
    def from_string(cls, student_str):
        name, grade = student_str.split('-')
        return cls(name, grade)

# Using class method to modify class variable
Student.change_school("United International University")
print(Student.school_name)  # Output: United International University

# Using factory method to create object
student = Student.from_string("Alice-A")
print(student.name)   # Output: Alice
print(student.grade)  # Output: A
```

### Static Methods

**Static methods** don't operate on instances or the class itself. They're essentially regular functions that happen to be defined inside a class because they have a logical relationship to the class. Static methods are defined using the `@staticmethod` decorator and don't receive any special first argument. They're useful for utility functions that relate to the class but don't need access to instance or class data.

```python
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius
    
    # Static method - utility function, no self or cls
    @staticmethod
    def celsius_to_fahrenheit(celsius):
        return (celsius * 9/5) + 32
    
    @staticmethod
    def fahrenheit_to_celsius(fahrenheit):
        return (fahrenheit - 32) * 5/9

# Called on the class - no instance needed
print(Temperature.celsius_to_fahrenheit(0))    # Output: 32.0
print(Temperature.fahrenheit_to_celsius(100))  # Output: 37.78

# Can also be called on an instance
temp = Temperature(25)
print(temp.celsius_to_fahrenheit(25))  # Output: 77.0
```

---

## Constructors and Destructors

### The Constructor: `__init__`

The **constructor** is a special method that is automatically called when you create a new object from a class. In Python, this method is named `__init__` (short for initialization). Its primary purpose is to initialize the object's attributes with values. When you write `Dog("Buddy", 3)`, Python first creates an empty object, then calls `__init__` with `self` set to the new object, along with the arguments you provided.

```python
class Book:
    def __init__(self, title, author, pages):
        # This runs automatically when creating a Book object
        print(f"Creating a new book: {title}")
        self.title = title
        self.author = author
        self.pages = pages
        self.current_page = 1  # Default value

# The __init__ method runs automatically
book = Book("Python Basics", "John Doe", 300)
# Output: Creating a new book: Python Basics

print(book.title)        # Output: Python Basics
print(book.current_page) # Output: 1
```

### The Destructor: `__del__`

The **destructor** is a special method called when an object is about to be destroyed (removed from memory). In Python, this method is named `__del__`. Destructors are less commonly used in Python than in other languages because Python has automatic garbage collection. However, they can be useful for cleanup tasks like closing files or releasing external resources.

```python
class FileManager:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'w')
        print(f"Opened file: {filename}")
    
    def write(self, content):
        self.file.write(content)
    
    # Destructor - cleanup when object is destroyed
    def __del__(self):
        self.file.close()
        print(f"Closed file: {self.filename}")

# Using the class
fm = FileManager("test.txt")
fm.write("Hello, World!")

# When fm goes out of scope or is deleted, __del__ is called
del fm  # Output: Closed file: test.txt
```

---

## Encapsulation and Access Control

### What is Encapsulation?

**Encapsulation** is the practice of bundling data (attributes) and the methods that operate on that data within a class, while restricting direct access to some of the object's components. Think of it as protective packaging around your data. Encapsulation prevents external code from directly modifying internal data, which could put your object in an inconsistent state. Instead, external code must use methods you provide, which can validate data and maintain consistency.

### Access Modifiers in Python

Unlike languages like Java or C++, Python doesn't have true access modifiers like `public`, `private`, or `protected`. Instead, Python uses naming conventions to indicate the intended visibility of attributes and methods:

* **Public**: Normal names like `name` or `age`. Accessible from anywhere.

* **Protected**: Names starting with a single underscore like `_name`. By convention, this signals that it's for internal use, but it can still be accessed from outside.

* **Private**: Names starting with double underscores like `__name`. Python applies name mangling to make it harder to access from outside.

```python
class BankAccount:
    def __init__(self, owner, balance):
        self.owner = owner           # Public attribute
        self._account_type = "Savings"  # Protected attribute (convention)
        self.__balance = balance     # Private attribute (name mangling)
    
    # Public method to access private data
    def get_balance(self):
        return self.__balance
    
    # Public method to modify private data (with validation)
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            return f"Deposited ${amount}. New balance: ${self.__balance}"
        return "Invalid amount"

# Using the class
account = BankAccount("Alice", 1000)

print(account.owner)          # Output: Alice (public - accessible)
print(account._account_type)  # Output: Savings (protected - accessible but discouraged)

# This would raise an error:
# print(account.__balance)  # AttributeError

# Access through public method
print(account.get_balance())  # Output: 1000
print(account.deposit(500))   # Output: Deposited $500. New balance: $1500
```

### Name Mangling

**Name mangling** is Python's mechanism for making attributes with double underscores more private. When you name an attribute with double underscores (like `__balance`), Python automatically renames it to `_ClassName__attribute` (like `_BankAccount__balance`). This doesn't make it truly private—determined programmers can still access it—but it prevents accidental access and name collisions in inheritance.

```python
class Secret:
    def __init__(self):
        self.__private_data = "Hidden"

s = Secret()

# Direct access fails
# print(s.__private_data)  # AttributeError

# But with mangled name, it works (though not recommended)
print(s._Secret__private_data)  # Output: Hidden

# You can see all attributes
print(s.__dict__)  # Output: {'_Secret__private_data': 'Hidden'}
```

---

## Abstraction and Abstract Classes

### What is Abstraction?

**Abstraction** means hiding complex implementation details and showing only the essential features to the user. When you drive a car, you interact with the steering wheel, pedals, and dashboard—you don't need to know exactly how the engine works internally. Similarly, in programming, abstraction lets users of your class interact with a simple interface without worrying about internal complexities. This makes code easier to use and maintain.

### Abstract Classes

An **abstract class** is a class that cannot be instantiated directly—it's meant to be a blueprint for other classes. Abstract classes can contain abstract methods, which are methods declared but not implemented. Subclasses must implement these abstract methods. This enforces a contract: any subclass must provide specific functionality. In Python, you create abstract classes using the `abc` module.

```python
from abc import ABC, abstractmethod

# Abstract class - cannot be instantiated directly
class Shape(ABC):
    def __init__(self, color):
        self.color = color
    
    # Abstract method - must be implemented by subclasses
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass
    
    # Concrete method - inherited by all subclasses
    def describe(self):
        return f"This is a {self.color} shape."


# Concrete class - implements all abstract methods
class Rectangle(Shape):
    def __init__(self, color, width, height):
        super().__init__(color)
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)


class Circle(Shape):
    def __init__(self, color, radius):
        super().__init__(color)
        self.radius = radius
    
    def area(self):
        import math
        return math.pi * self.radius ** 2
    
    def perimeter(self):
        import math
        return 2 * math.pi * self.radius


# Cannot instantiate abstract class
# shape = Shape("red")  # TypeError

# Can instantiate concrete subclasses
rect = Rectangle("blue", 5, 3)
circle = Circle("green", 4)

print(rect.describe())      # Output: This is a blue shape.
print(f"Rectangle area: {rect.area()}")  # Output: Rectangle area: 15
print(f"Circle area: {circle.area():.2f}")  # Output: Circle area: 50.27
```

---

## Inheritance

### What is Inheritance?

**Inheritance** is a mechanism where a new class (child class or subclass) acquires the attributes and methods of an existing class (parent class or base class). This promotes code reuse and establishes a relationship between classes. Think of it like genetic inheritance: a child inherits traits from their parents but can also have unique characteristics. In programming, a subclass inherits functionality from its parent but can add new features or modify existing ones.

### Base Class and Subclass

The **base class** (also called parent or superclass) is the class being inherited from. The **subclass** (also called child or derived class) is the class that inherits from the base class. A subclass automatically has access to all public and protected attributes and methods of its base class.

```python
# Base class (Parent)
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return "Some sound"
    
    def eat(self):
        return f"{self.name} is eating"


# Subclass (Child) - inherits from Animal
class Dog(Animal):
    def speak(self):  # Override parent method
        return "Woof!"
    
    def fetch(self):  # New method specific to Dog
        return f"{self.name} is fetching the ball"


# Another subclass
class Cat(Animal):
    def speak(self):  # Override parent method
        return "Meow!"
    
    def scratch(self):  # New method specific to Cat
        return f"{self.name} is scratching"


# Creating objects
dog = Dog("Buddy")
cat = Cat("Whiskers")

print(dog.name)       # Output: Buddy (inherited attribute)
print(dog.eat())      # Output: Buddy is eating (inherited method)
print(dog.speak())    # Output: Woof! (overridden method)
print(dog.fetch())    # Output: Buddy is fetching the ball (new method)

print(cat.speak())    # Output: Meow! (overridden method)
print(cat.scratch())  # Output: Whiskers is scratching (new method)
```

### The `super()` Function

The **`super()`** function allows a subclass to call methods from its parent class. This is especially useful in the `__init__` method when you want to extend the parent's initialization rather than completely replace it. Using `super()` ensures that the parent class is properly initialized, which is crucial for maintaining the inheritance chain.

```python
class Vehicle:
    def __init__(self, brand, year):
        self.brand = brand
        self.year = year
    
    def info(self):
        return f"{self.brand} ({self.year})"


class Car(Vehicle):
    def __init__(self, brand, year, model, doors):
        # Call parent's __init__ method
        super().__init__(brand, year)
        # Add Car-specific attributes
        self.model = model
        self.doors = doors
    
    def info(self):
        # Extend parent's info method
        base_info = super().info()
        return f"{base_info} - {self.model}, {self.doors} doors"


car = Car("Toyota", 2023, "Camry", 4)
print(car.info())  # Output: Toyota (2023) - Camry, 4 doors
```

### Types of Inheritance

Python supports multiple types of inheritance patterns:

* **Single Inheritance**: A subclass inherits from one parent class. This is the most common and simplest form.

* **Multiple Inheritance**: A subclass inherits from multiple parent classes. This can be powerful but also complex.

* **Multilevel Inheritance**: A subclass inherits from a parent, which itself inherits from another parent, forming a chain.

* **Hierarchical Inheritance**: Multiple subclasses inherit from a single parent class.

```python
# Multiple Inheritance example
class Flyable:
    def fly(self):
        return "Flying high!"


class Swimmable:
    def swim(self):
        return "Swimming in water!"


# Duck inherits from both Flyable and Swimmable
class Duck(Flyable, Swimmable):
    def quack(self):
        return "Quack!"


duck = Duck()
print(duck.fly())   # Output: Flying high!
print(duck.swim())  # Output: Swimming in water!
print(duck.quack()) # Output: Quack!
```

---

## Polymorphism

### What is Polymorphism?

**Polymorphism** comes from Greek words meaning "many forms." In OOP, polymorphism allows objects of different classes to be treated as objects of a common base class. The same operation can behave differently for different types of objects. This makes your code more flexible and extensible. When you call a method on an object, Python determines which method to call based on the object's actual type, not just the variable's declared type.

### Method Overriding

**Method overriding** occurs when a subclass provides a specific implementation of a method that is already defined in its parent class. This is how you customize inherited behavior. When you call the method on a subclass object, Python uses the subclass's version instead of the parent's version.

```python
class PaymentMethod:
    def process_payment(self, amount):
        return f"Processing payment of ${amount}"


class CreditCard(PaymentMethod):
    def process_payment(self, amount):
        return f"Charging ${amount} to credit card"


class PayPal(PaymentMethod):
    def process_payment(self, amount):
        return f"Processing ${amount} through PayPal"


class Bitcoin(PaymentMethod):
    def process_payment(self, amount):
        return f"Transferring ${amount} worth of Bitcoin"


# Polymorphism in action - same method, different behavior
methods = [CreditCard(), PayPal(), Bitcoin()]

for method in methods:
    print(method.process_payment(100))

# Output:
# Charging $100 to credit card
# Processing $100 through PayPal
# Transferring $100 worth of Bitcoin
```

### Operator Overloading

**Operator overloading** allows you to define how operators like `+`, `-`, `*`, `==`, and others behave when used with objects of your class. This is done by implementing special methods (also called **dunder methods** or magic methods) that Python calls when it encounters operators. For example, implementing `__add__` defines how the `+` operator works for your objects.

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # Overloading the + operator
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    # Overloading the - operator
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    
    # Overloading the == operator
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    # String representation
    def __str__(self):
        return f"Vector({self.x}, {self.y})"


v1 = Vector(3, 4)
v2 = Vector(1, 2)

# Using overloaded operators
v3 = v1 + v2
print(v3)        # Output: Vector(4, 6)

v4 = v1 - v2
print(v4)        # Output: Vector(2, 2)

print(v1 == Vector(3, 4))  # Output: True
```

### Common Dunder Methods

Dunder methods (double underscore methods) let you define how your objects behave with Python's built-in operations:

| Method | Purpose | Example |
|--------|---------|---------|
| `__str__` | String representation (for `print()`) | `print(obj)` |
| `__repr__` | Official string representation | `repr(obj)` |
| `__len__` | Length of object | `len(obj)` |
| `__getitem__` | Index access | `obj[key]` |
| `__setitem__` | Index assignment | `obj[key] = value` |
| `__contains__` | Membership test | `item in obj` |
| `__iter__` | Iteration support | `for item in obj` |

```python
class ShoppingCart:
    def __init__(self):
        self.items = []
    
    def add(self, item, price):
        self.items.append((item, price))
    
    def __len__(self):
        return len(self.items)
    
    def __str__(self):
        total = sum(price for _, price in self.items)
        items_str = "\n".join(f"  - {item}: ${price}" for item, price in self.items)
        return f"Shopping Cart:\n{items_str}\nTotal: ${total}"
    
    def __contains__(self, item_name):
        return any(item == item_name for item, _ in self.items)
    
    def __getitem__(self, index):
        return self.items[index]


cart = ShoppingCart()
cart.add("Apple", 1.50)
cart.add("Bread", 2.00)
cart.add("Milk", 3.50)

print(len(cart))        # Output: 3
print("Apple" in cart)  # Output: True
print(cart[0])          # Output: ('Apple', 1.5)
print(cart)             # Prints formatted cart
```

---

## Named Tuples and Data Classes

### Named Tuples

**Named tuples** are a memory-efficient way to create simple classes that primarily store data. They're like regular tuples, but each element has a name, making your code more readable. Named tuples are immutable (cannot be changed after creation) and are perfect for representing simple data structures like coordinates, RGB colors, or database records.

```python
from collections import namedtuple

# Creating a named tuple type
Point = namedtuple('Point', ['x', 'y'])
Person = namedtuple('Person', 'name age city')

# Creating instances
p1 = Point(3, 4)
p2 = Point(x=10, y=20)

person = Person("Alice", 25, "New York")

# Accessing by name or index
print(p1.x, p1.y)      # Output: 3 4
print(p1[0], p1[1])    # Output: 3 4

print(person.name)     # Output: Alice
print(person.age)      # Output: 25

# Named tuples are immutable
# p1.x = 5  # Would raise AttributeError

# Useful methods
print(p1._asdict())    # Output: {'x': 3, 'y': 4}
print(p1._replace(x=5))  # Output: Point(x=5, y=4)
```

### Data Classes

**Data classes** (introduced in Python 3.7) provide a more modern way to create classes that primarily store data. They automatically generate useful methods like `__init__`, `__repr__`, and `__eq__` for you, reducing boilerplate code. Data classes can be mutable or immutable and support default values, type hints, and inheritance.

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Student:
    name: str
    student_id: int
    grades: List[int] = field(default_factory=list)
    gpa: float = 0.0
    
    def add_grade(self, grade):
        self.grades.append(grade)
        self.calculate_gpa()
    
    def calculate_gpa(self):
        if self.grades:
            self.gpa = sum(self.grades) / len(self.grades)


# Creating instances
student1 = Student("Alice", 1001)
student2 = Student("Bob", 1002, [85, 90], 3.5)

# Auto-generated __repr__
print(student1)  # Output: Student(name='Alice', student_id=1001, grades=[], gpa=0.0)

# Adding grades
student1.add_grade(95)
student1.add_grade(88)
print(student1)  # Output: Student(name='Alice', student_id=1001, grades=[95, 88], gpa=91.5)

# Auto-generated __eq__
student3 = Student("Alice", 1001)
print(student1 == student3)  # Output: False (different grades)
```

---

## Exception Handling

### Understanding Exceptions

**Exceptions** are errors that occur during program execution. Unlike syntax errors, which are detected before the program runs, exceptions happen while the program is running. Python uses exceptions to signal that something unexpected happened—a file wasn't found, a division by zero was attempted, or an index was out of range. Exception handling allows you to gracefully respond to these errors instead of letting your program crash.

### Try-Except Blocks

The basic structure for handling exceptions is the **try-except** block. You put code that might raise an exception in the `try` block, and code to handle the exception in the `except` block. This prevents your program from crashing and allows you to provide meaningful error messages or take corrective action.

```python
# Basic exception handling
try:
    number = int(input("Enter a number: "))
    result = 10 / number
    print(f"10 divided by {number} is {result}")
except ValueError:
    print("That's not a valid number!")
except ZeroDivisionError:
    print("You can't divide by zero!")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Handling multiple exceptions together
try:
    data = [1, 2, 3]
    index = int(input("Enter index: "))
    print(data[index])
except (ValueError, IndexError) as e:
    print(f"Error: {e}")
```

### Raising Exceptions

You can also **raise** exceptions in your own code when you detect an error condition. This is useful for validating inputs and enforcing rules in your classes.

```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
    
    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount
        return self.balance


# Using the class with exception handling
account = BankAccount("Alice", 100)

try:
    account.withdraw(150)
except ValueError as e:
    print(f"Transaction failed: {e}")  # Output: Transaction failed: Insufficient funds
```

### Finally and Else Clauses

The **finally** block always executes, whether an exception occurred or not. It's commonly used for cleanup operations like closing files. The **else** block executes only if no exception was raised in the try block.

```python
def read_file(filename):
    try:
        file = open(filename, 'r')
        content = file.read()
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None
    else:
        print("File read successfully")
        return content
    finally:
        print("Cleanup complete")
        try:
            file.close()
        except:
            pass
```

### Custom Exceptions

You can create your own exception classes by inheriting from Python's built-in `Exception` class. This allows you to create meaningful exception types specific to your application.

```python
class InsufficientFundsError(Exception):
    """Exception raised when account has insufficient funds."""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        self.message = f"Cannot withdraw ${amount}. Balance: ${balance}"
        super().__init__(self.message)


class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
    
    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientFundsError(self.balance, amount)
        self.balance -= amount
        return self.balance


# Using custom exception
account = BankAccount("Alice", 100)

try:
    account.withdraw(150)
except InsufficientFundsError as e:
    print(f"Error: {e}")  # Output: Error: Cannot withdraw $150. Balance: $100
```

---

## Context Managers

### What is a Context Manager?

A **context manager** is a convenient way to manage resources that need to be set up and cleaned up, like files, network connections, or database sessions. The most common way to use context managers is with the `with` statement, which automatically handles setup and cleanup even if an error occurs. This ensures resources are properly released, preventing memory leaks and locked files.

### Creating Context Managers

You can create context managers in two ways: using a class with `__enter__` and `__exit__` methods, or using a generator function with the `@contextmanager` decorator.

```python
# Context manager using a class
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        # Setup: open the file
        print(f"Opening {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup: close the file
        print(f"Closing {self.filename}")
        if self.file:
            self.file.close()
        # Return False to propagate exceptions, True to suppress them
        return False


# Using the context manager
with FileManager("test.txt", "w") as f:
    f.write("Hello, World!")
# File is automatically closed when exiting the block
```

```python
# Context manager using a generator function
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    start = time.time()
    print("Timer started")
    
    yield  # Code inside 'with' block runs here
    
    end = time.time()
    print(f"Timer ended. Elapsed: {end - start:.2f} seconds")


# Using the context manager
with timer():
    total = sum(range(1000000))
    print(f"Sum calculated: {total}")
# Output shows elapsed time automatically
```

---

## Generators

### What is a Generator?

A **generator** is a special type of function that produces a sequence of values one at a time, instead of computing and returning all values at once. This is incredibly memory-efficient when working with large datasets or infinite sequences. Instead of building a list with millions of items in memory, a generator produces each value only when it's requested. Generators use the `yield` keyword instead of `return`, which allows them to pause and resume execution.

### Creating Generators

```python
# A simple generator function
def count_up_to(n):
    """Generator that yields numbers from 1 to n."""
    count = 1
    while count <= n:
        yield count  # Pause here and return value
        count += 1   # Resume here on next call


# Using the generator
counter = count_up_to(5)

print(next(counter))  # Output: 1
print(next(counter))  # Output: 2
print(next(counter))  # Output: 3

# Can iterate over remaining values
for num in counter:
    print(num)  # Output: 4, 5

# Generator is now exhausted
# next(counter)  # Would raise StopIteration
```

### Generator Expressions

Generator expressions provide a concise syntax for creating generators, similar to list comprehensions but with parentheses instead of brackets.

```python
# List comprehension (creates entire list in memory)
squares_list = [x**2 for x in range(1000000)]

# Generator expression (creates values on demand)
squares_gen = (x**2 for x in range(1000000))

print(f"List size: {squares_list.__sizeof__()} bytes")
print(f"Generator size: {squares_gen.__sizeof__()} bytes")

# Generator can be used in iterations
for i, square in enumerate(squares_gen):
    if i >= 5:
        break
    print(square)  # Output: 0, 1, 4, 9, 16
```

### Practical Example: Reading Large Files

Generators are perfect for processing large files line by line without loading the entire file into memory.

```python
def read_large_file(filename):
    """Generator to read a large file line by line."""
    with open(filename, 'r') as file:
        for line in file:
            yield line.strip()


def process_large_file(filename):
    """Process a large file efficiently using a generator."""
    line_count = 0
    word_count = 0
    
    for line in read_large_file(filename):
        line_count += 1
        word_count += len(line.split())
    
    return line_count, word_count


# Memory-efficient file processing
# lines, words = process_large_file("large_file.txt")
```

---

## Decorators

### What is a Decorator?

A **decorator** is a function that modifies the behavior of another function without changing its source code. Think of it as wrapping a gift: the gift (function) stays the same, but you add decorative paper (the decorator) that changes how it's presented. Decorators are widely used in Python for logging, timing functions, checking permissions, caching results, and many other cross-cutting concerns. They follow the principle of "don't repeat yourself" by letting you write common functionality once and apply it to many functions.

### Creating Decorators

```python
# A simple decorator
def my_decorator(func):
    def wrapper():
        print("Something happens before the function")
        func()  # Call the original function
        print("Something happens after the function")
    return wrapper


# Applying the decorator
@my_decorator
def say_hello():
    print("Hello!")


# Calling the decorated function
say_hello()
# Output:
# Something happens before the function
# Hello!
# Something happens after the function
```

### Decorators with Arguments

To handle functions with arguments, your wrapper function needs to accept and pass along any arguments.

```python
def timer(func):
    """Decorator to measure function execution time."""
    import time
    
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)  # Call with any arguments
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    
    return wrapper


@timer
def slow_function(n):
    """A function that takes some time."""
    total = 0
    for i in range(n):
        total += i ** 2
    return total


# The decorator automatically times the function
result = slow_function(1000000)
# Output: slow_function took 0.1234 seconds
```

### Practical Decorator Examples

```python
def retry(attempts=3, delay=1):
    """Decorator to retry a function if it fails."""
    import time
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < attempts - 1:
                        print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                        time.sleep(delay)
                    else:
                        print(f"All {attempts} attempts failed.")
                        raise
        return wrapper
    return decorator


@retry(attempts=3, delay=0.5)
def fetch_data(url):
    """Simulated function that might fail."""
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("Network error")
    return f"Data from {url}"


# The function will be retried automatically
# data = fetch_data("https://example.com/api")
```

```python
def log_function(func):
    """Decorator to log function calls."""
    def wrapper(*args, **kwargs):
        args_str = ", ".join(repr(a) for a in args)
        kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
        all_args = ", ".join(filter(None, [args_str, kwargs_str]))
        
        print(f"Calling {func.__name__}({all_args})")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {repr(result)}")
        
        return result
    return wrapper


@log_function
def add(a, b):
    return a + b


@log_function
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"


# All calls are automatically logged
add(3, 5)
# Output:
# Calling add(3, 5)
# add returned 8

greet("Alice", greeting="Hi")
# Output:
# Calling greet('Alice', greeting='Hi')
# greet returned 'Hi, Alice!'
```

---

## Putting It All Together

### A Complete Example: Library Management System

Let's create a comprehensive example that demonstrates multiple OOP concepts working together:

```python
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
from abc import ABC, abstractmethod


# Custom exception
class BookNotAvailableError(Exception):
    """Raised when trying to borrow an unavailable book."""
    pass


# Abstract base class
class Item(ABC):
    def __init__(self, title, item_id):
        self.title = title
        self.item_id = item_id
        self.is_available = True
    
    @abstractmethod
    def get_info(self):
        pass
    
    def __str__(self):
        return f"{self.title} ({'Available' if self.is_available else 'Borrowed'})"


# Data class for simplicity
@dataclass
class Author:
    name: str
    nationality: str
    
    def __str__(self):
        return f"{self.name} ({self.nationality})"


# Concrete class inheriting from Item
class Book(Item):
    def __init__(self, title, item_id, author: Author, isbn: str):
        super().__init__(title, item_id)
        self.author = author
        self.isbn = isbn
    
    def get_info(self):
        return f"'{self.title}' by {self.author.name} (ISBN: {self.isbn})"


# Another concrete class
class Magazine(Item):
    def __init__(self, title, item_id, issue_number: int):
        super().__init__(title, item_id)
        self.issue_number = issue_number
    
    def get_info(self):
        return f"'{self.title}' - Issue #{self.issue_number}"


# Class demonstrating encapsulation and private attributes
class Member:
    def __init__(self, name, member_id):
        self.name = name
        self.__member_id = member_id  # Private
        self.__borrowed_items: List[Item] = []  # Private
    
    @property
    def member_id(self):
        return self.__member_id
    
    def borrow_item(self, item: Item):
        if not item.is_available:
            raise BookNotAvailableError(f"'{item.title}' is not available")
        item.is_available = False
        self.__borrowed_items.append(item)
        return f"{self.name} borrowed {item.title}"
    
    def return_item(self, item: Item):
        if item in self.__borrowed_items:
            item.is_available = True
            self.__borrowed_items.remove(item)
            return f"{self.name} returned {item.title}"
        return f"{self.name} doesn't have this item"
    
    def get_borrowed_items(self):
        return [item.title for item in self.__borrowed_items]


# Context manager for borrowing operations
class BorrowingSession:
    def __init__(self, member: Member, item: Item):
        self.member = member
        self.item = item
    
    def __enter__(self):
        print(f"Starting borrowing session for {self.member.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            print(f"Transaction failed: {exc_val}")
        else:
            print(f"Transaction completed successfully")
        return False


# Decorator for logging
def log_transaction(func):
    def wrapper(*args, **kwargs):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Executing {func.__name__}")
        result = func(*args, **kwargs)
        print(f"[{timestamp}] Completed {func.__name__}")
        return result
    return wrapper


# Main library class
class Library:
    def __init__(self, name: str):
        self.name = name
        self.__items: List[Item] = []  # Private
        self.__members: List[Member] = []  # Private
    
    @log_transaction
    def add_item(self, item: Item):
        self.__items.append(item)
        return f"Added: {item.get_info()}"
    
    @log_transaction
    def register_member(self, member: Member):
        self.__members.append(member)
        return f"Registered: {member.name}"
    
    def find_item(self, title: str) -> Optional[Item]:
        for item in self.__items:
            if title.lower() in item.title.lower():
                return item
        return None
    
    def list_available_items(self):
        return [item for item in self.__items if item.is_available]
    
    def __len__(self):
        return len(self.__items)
    
    def __contains__(self, title: str):
        return any(title.lower() in item.title.lower() for item in self.__items)


# Demo usage
if __name__ == "__main__":
    # Create library
    library = Library("UIU Central Library")
    
    # Create authors and books
    author1 = Author("Python Expert", "USA")
    author2 = Author("Java Guru", "UK")
    
    book1 = Book("Python Programming", "B001", author1, "978-0-123456-78-9")
    book2 = Book("Advanced Python", "B002", author1, "978-0-123456-79-6")
    magazine = Magazine("Tech Today", "M001", 42)
    
    # Add items to library
    print(library.add_item(book1))
    print(library.add_item(book2))
    print(library.add_item(magazine))
    
    # Register members
    member1 = Member("Alice", "M001")
    member2 = Member("Bob", "M002")
    
    print(library.register_member(member1))
    print(library.register_member(member2))
    
    # Borrow items using context manager
    with BorrowingSession(member1, book1):
        print(member1.borrow_item(book1))
    
    # Try to borrow same book (will fail)
    try:
        member2.borrow_item(book1)
    except BookNotAvailableError as e:
        print(f"Error: {e}")
    
    # Check available items
    print(f"\nAvailable items: {[item.title for item in library.list_available_items()]}")
    
    # Return book
    print(member1.return_item(book1))
    
    # Use membership test
    print(f"\n'Python' in library: {'Python' in library}")
```

---

## Conclusion

Object-Oriented Programming in Python provides a powerful and intuitive way to structure your code. Throughout this guide, we've explored the fundamental concepts that form the foundation of OOP: classes and objects for organizing data and behavior, encapsulation for protecting your data, inheritance for code reuse, and polymorphism for flexibility. We've also covered advanced concepts like abstract classes, decorators, generators, and context managers that make Python a uniquely expressive language.

The journey from writing simple procedural scripts to designing elegant object-oriented systems is transformative. As you practice these concepts, you'll find that OOP helps you write code that is not only more organized but also easier to maintain, test, and extend. Remember that good OOP design comes with practice—start by identifying objects in the problems you're trying to solve, define their attributes and behaviors, and gradually build up to more complex class hierarchies and design patterns.

Whether you're building a simple command-line tool, a web application, or a data analysis pipeline, the OOP principles covered in this guide will serve you well. The examples provided here are starting points; experiment with them, modify them, and apply these concepts to your own projects. As you continue your programming journey at UIU and beyond, mastering OOP will be one of the most valuable skills in your toolkit.
