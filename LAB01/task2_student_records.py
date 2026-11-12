students = {
    101: {"Name": "Ali",   "Major": "AI",       "Grades": [85, 90, 78]},
    102: {"Name": "Sara",  "Major": "CS",       "Grades": [92, 88, 95]},
    103: {"Name": "Hamza", "Major": "AI",       "Grades": [70, 65, 80]},
    104: {"Name": "Ayesha", "Major": "Software", "Grades": [88, 91, 84]},
}


def top_student(records):
    """Return the name of the student with the highest average grade."""
    best_name, best_avg = None, float("-inf")
    for info in records.values():
        if not info["Grades"]:
            continue
        avg = sum(info["Grades"]) / len(info["Grades"])
        if avg > best_avg:
            best_name, best_avg = info["Name"], avg
    return best_name


def search_by_major(records, major):
    """Print all students enrolled in the given major."""
    found = [(sid, i["Name"]) for sid, i in records.items()
             if i["Major"].lower() == major.lower()]
    if not found:
        print(f"No students found in {major}.")
        return
    print(f"Students in {major}:")
    for sid, name in found:
        print(f"  {sid}: {name}")


if __name__ == "__main__":
    print("Top student:", top_student(students))
    search_by_major(students, "AI")
    search_by_major(students, "Math")
