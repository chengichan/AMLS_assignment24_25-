
# Import the modules from the respective folders
from A import task_A
from B import task_B

def run_tasks():
    while True:
        print("\nSelect a task to run:")
        print("A: Task A")
        print("B: Task B")
        print("Q: Quit")
        
        choice = input("Enter your choice (A/B/Q): ").strip().upper()
        
        if choice == 'A':
            print("Running Task A...\n")
            task_A.main()
        elif choice == 'B':
            print("Running Task B...\n")
            task_B.main()
        elif choice == 'Q':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter A, B, or Q.")

if __name__ == '__main__':
    run_tasks()
