class GroceryManager:
    def __init__(self):
        self.items = {}   # name -> {"quantity": int, "price": float}

    def add_item(self, item, quantity, price):
        if quantity <= 0 or price < 0:
            raise ValueError("Quantity must be positive and price cannot be negative.")
        if item in self.items:
            self.items[item]["quantity"] += quantity
            self.items[item]["price"] = price
        else:
            self.items[item] = {"quantity": quantity, "price": price}
        print(f"Added {quantity} x {item} @ {price}")

    def remove_item(self, item):
        if item not in self.items:
            print(f"Error: '{item}' is not in the list.")
            return False
        del self.items[item]
        print(f"Removed {item}")
        return True

    def view_list(self):
        if not self.items:
            print("Grocery list is empty.")
            return
        print(f"{'Item':<15}{'Qty':>5}{'Price':>10}{'Subtotal':>12}")
        for name, d in self.items.items():
            print(f"{name:<15}{d['quantity']:>5}{d['price']:>10.2f}{d['quantity'] * d['price']:>12.2f}")

    def calculate_total(self):
        return sum(d["quantity"] * d["price"] for d in self.items.values())


if __name__ == "__main__":
    g = GroceryManager()
    g.add_item("Milk", 2, 250.0)
    g.add_item("Bread", 1, 120.0)
    g.add_item("Eggs", 12, 30.0)
    g.view_list()
    print("Total:", g.calculate_total())
    g.remove_item("Bread")
    g.remove_item("Butter")      # does not exist -> handled
    try:
        g.add_item("Rice", -1, 100)
    except ValueError as e:
        print("Error:", e)
    g.view_list()
    print("Total:", g.calculate_total())
