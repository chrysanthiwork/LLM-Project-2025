import subprocess
import sys # Για να πάρουμε τη διαδρομή του python interpreter

def run_script(script_path):
    """Εκτελεί ένα Python script χρησιμοποιώντας το subprocess."""
    try:
        # Χρησιμοποιούμε το sys.executable για να είμαστε σίγουροι ότι τρέχει με τον ίδιο interpreter
        print(f"--- Ξεκινά η εκτέλεση του {script_path} ---")
        result = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True)
        print(f"Έξοδος από {script_path}:\n{result.stdout}")
        if result.stderr:
            print(f"Σφάλματα από {script_path}:\n{result.stderr}")
        print(f"--- Ολοκληρώθηκε η εκτέλεση του {script_path} ---")
        return True
    except FileNotFoundError:
        print(f"Σφάλμα: Το αρχείο {script_path} δεν βρέθηκε.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Σφάλμα κατά την εκτέλεση του {script_path}.")
        print(f"Έξοδος σφάλματος:\n{e.stderr}")
        return False

# Articles is the first file that must be executed and after that, the Crawler one is necessary for our code to work. Finally we must execute the preprocessing one.
def main():
    scripts_to_run = [
        "representation.py",
        "similarities.py"
    ]

    print("Ξεκινά η εκτέλεση των scripts με τη σειρά...")
    for script_file in scripts_to_run:
        print(f"\nΠροσπάθεια εκτέλεσης: {script_file}")
        if not run_script(script_file):
            print(f"Η εκτέλεση του {script_file} απέτυχε. Διακοπή.")
            break # Σταμάτα αν κάποιο script αποτύχει
    print("\nΌλα τα επιλεγμένα scripts (ή μέχρι το σημείο αποτυχίας) έχουν εκτελεστεί.")

if __name__ == "__main__":
    main()