Below is a structured guide for **Day 2, Week 1**. The focus is to make you comfortable with **core Python** inside Jupyter so that later ML code feels natural.

You can complete this in one or two sittings.

---

# Day 2 – Python Refresher & Basic Scripting in Jupyter

**Goal:**  
Refresh/learn core Python constructs (variables, types, lists, dicts, loops, conditionals, functions) and practice them in a Jupyter notebook.

Approx. time: 1.5–2 hours.

---

## 0. Pre‑Work: Open Your Environment

1. **Activate your ML environment** (if not already):

   ```bash
   conda activate ml
   ```

2. **Go to your working folder** (same one as Day 1):

   ```bash
   cd path/to/your/ml-learning
   ```

3. **Launch JupyterLab** (or notebook):

   ```bash
   jupyter lab
   ```
   or
   ```bash
   jupyter notebook
   ```

4. **Create a new notebook**:
   - Name it: `day2_python_basics.ipynb`.

---

## 1. Variables, Basic Types, and Printing

**Objective:** Understand how Python stores values and how to inspect them.

### 1.1. Create and inspect variables

In your notebook:

```python
# numbers
x = 10            # int
y = 3.5           # float

# text
name = "Alice"    # string

# boolean
is_active = True  # bool

print("x:", x, "type:", type(x))
print("y:", y, "type:", type(y))
print("name:", name, "type:", type(name))
print("is_active:", is_active, "type:", type(is_active))
```

### 1.2. Basic operations

```python
a = 7
b = 3

print("a + b =", a + b)
print("a - b =", a - b)
print("a * b =", a * b)
print("a / b =", a / b)
print("a // b =", a // b)  # integer division
print("a % b =", a % b)    # remainder
print("a ** b =", a ** b)  # exponent
```

### Mini Exercise 1 (5 minutes)

Without copying:

1. Create two variables `length = 5` and `width = 2.5`.
2. Compute `area = length * width`.
3. Print: `"Area is: <area_value>"`.

*Check after you try:*

```python
length = 5
width = 2.5
area = length * width
print("Area is:", area)
```

---

## 2. Lists – Ordered Collections

**Objective:** Learn how to store and manipulate sequences of values.

### 2.1. Creating and accessing lists

```python
nums = [1, 2, 3, 4, 5]
print("nums:", nums)
print("First element:", nums[0])
print("Last element:", nums[-1])
print("Length:", len(nums))
```

### 2.2. Modifying lists

```python
nums.append(6)             # add to end
print("After append:", nums)

nums.insert(0, 0)          # insert at position 0
print("After insert at 0:", nums)

removed = nums.pop()       # remove last element
print("Popped:", removed)
print("After pop:", nums)
```

### 2.3. Slicing

```python
print("First three elements:", nums[:3])
print("From index 2 to 4:", nums[2:5])
print("Every second element:", nums[::2])
```

### Mini Exercise 2 (5–10 minutes)

1. Create a list: `fruits = ["apple", "banana", "cherry"]`.
2. Add `"orange"` to the end.
3. Replace `"banana"` with `"grape"`.
4. Print the final list and its length.

*Possible solution (only check after trying):*

```python
fruits = ["apple", "banana", "cherry"]
fruits.append("orange")
fruits[1] = "grape"
print(fruits)
print("Length:", len(fruits))
```

---

## 3. Dictionaries – Key/Value Mappings

**Objective:** Store and access data by name (key) instead of position.

### 3.1. Creating and accessing dictionaries

```python
person = {
    "name": "Bob",
    "age": 28,
    "city": "London"
}

print(person)
print("Name:", person["name"])
print("Age:", person["age"])
```

### 3.2. Adding and updating entries

```python
person["job"] = "Engineer"          # add new key
print(person)

person["age"] = 29                  # update existing key
print("Updated age:", person["age"])
```

### 3.3. Iterating over dictionaries

```python
for key, value in person.items():
    print(key, "->", value)
```

### Mini Exercise 3 (5–10 minutes)

1. Create a dict `movie` with keys: `"title"`, `"year"`, `"rating"`.
2. Print the title and rating.
3. Add a new key `"genre"` with any value.
4. Loop over keys and values and print them.

---

## 4. Control Flow – `if` Statements

**Objective:** Make decisions in your code.

### 4.1. Basic `if / elif / else`

```python
n = 7

if n > 10:
    print("n is greater than 10")
elif n == 10:
    print("n is equal to 10")
else:
    print("n is less than 10")
```

### 4.2. Combining conditions

```python
age = 20
has_id = True

if age >= 18 and has_id:
    print("Allowed entry")
else:
    print("Not allowed")
```

### Mini Exercise 4 (5–10 minutes)

Write code that:

1. Has a numeric variable `score`.
2. Prints:
   - `"Excellent"` if score ≥ 90
   - `"Good"` if 70 ≤ score < 90
   - `"Needs improvement"` otherwise

---

## 5. Loops – `for` and `while`

**Objective:** Repeat actions over data or while a condition holds.

### 5.1. `for` loops over a range

```python
for i in range(5):
    print("i is:", i)
```

### 5.2. `for` loops over lists

```python
fruits = ["apple", "banana", "cherry"]

for fruit in fruits:
    print("Fruit:", fruit)
```

### 5.3. `while` loops

```python
count = 0
while count < 3:
    print("Count is:", count)
    count += 1
```

### Mini Exercise 5 (10 minutes)

1. Use a `for` loop to compute the sum of numbers 1 to 10.
   - Hint: initialize `total = 0`, then loop and add.

2. Use a `while` loop to count down from 5 to 1 and then print `"Lift off!"`.

---

## 6. Functions – Reusing Logic

**Objective:** Group code into reusable blocks.

### 6.1. Define simple functions

```python
def square(x):
    return x * x

print(square(3))
print(square(10))
```

### 6.2. Functions with multiple parameters

```python
def describe_person(name, age):
    return f"{name} is {age} years old."

print(describe_person("Alice", 30))
print(describe_person("Bob", 25))
```

### 6.3. Default arguments (optional but useful)

```python
def greet(name="there"):
    print(f"Hello, {name}!")

greet("Alice")
greet()  # uses default
```

### Mini Exercise 6 – Mean of a List (Important) (10–15 minutes)

Write a function `mean_of_list(lst)` that:

- Takes a list of numbers.
- Returns the arithmetic mean (sum divided by length).

Steps:

1. Check that `lst` is not empty (optional but good practice).
2. Use a loop or `sum(lst)` and `len(lst)`.
3. Test on `[1, 2, 3, 4]` and `[10, 20, 30]`.

*Possible solution (after trying):*

```python
def mean_of_list(lst):
    if len(lst) == 0:
        return None  # or raise an error
    return sum(lst) / len(lst)

print(mean_of_list([1, 2, 3, 4]))     # 2.5
print(mean_of_list([10, 20, 30]))     # 20.0
```

---

## 7. Small “Putting It Together” Exercise

**Objective:** Combine lists, loops, conditionals, and functions in a small task.

### Task: Student Grades Summary

1. Create a list of dictionaries, where each dict is a student:

   ```python
   students = [
       {"name": "Alice", "score": 88},
       {"name": "Bob", "score": 72},
       {"name": "Charlie", "score": 95},
       {"name": "Diana", "score": 60}
   ]
   ```

2. Write a function `grade(score)` that returns:
   - `"A"` if score ≥ 90
   - `"B"` if 80 ≤ score < 90
   - `"C"` if 70 ≤ score < 80
   - `"D"` if 60 ≤ score < 70
   - `"F"` otherwise

3. Loop over students, and for each:
   - Compute their grade using `grade(score)`.
   - Print something like: `"Alice scored 88 and got grade B"`.

4. (Optional) Compute and print:
   - The class average score.
   - Number of students with grade `"A"`.

This exercise mirrors how you’ll later loop over rows in a dataset and compute derived values.

---

## 8. Organize & Save Your Notebook

Before finishing:

1. Scroll through `day2_python_basics.ipynb` and:
   - Add **markdown headings** for each section:
     - `# Variables and Types`
     - `# Lists`
     - `# Dictionaries`
     - `# Control Flow`
     - `# Loops`
     - `# Functions`
     - `# Mini Project: Student Grades`
2. Make sure each code cell runs without errors from top to bottom:
   - Use **Kernel → Restart & Run All** to verify.
3. Save the notebook:
   - `File → Save` or `Ctrl + S` / `Cmd + S`.

---

## 9. Quick Reflection (3–5 minutes)

In a final markdown cell, answer briefly:

- Which Python concept felt most natural today?
- Which concept (e.g., loops, dicts, functions) do you want to practice again?
- Any questions/confusions to revisit later?

This will guide your review at the end of the week.

---

## Day 2 Completion Checklist

By the end of Day 2, you should be able to say “yes” to:

- [ ] I can create and manipulate basic Python types (int, float, string, bool).
- [ ] I can create a list, append/modify elements, and iterate over it.
- [ ] I can create and use dictionaries (key/value pairs).
- [ ] I can use `if / elif / else` to control program flow.
- [ ] I can write and use both `for` and `while` loops.
- [ ] I can define and call simple functions that take parameters and return values.
- [ ] I completed a small “student grades” mini‑project combining these concepts.

If you’d like, you can paste your `grade(score)` function or your mini‑project code, and I can review it and suggest Pythonic improvements.
