import subprocess
import traci
 
def main():
    print("=" * 50)
    print("NeuroFlow-FL Docker environment check")
    print("=" * 50)
 
    # Check SUMO binary is available and print its version
    result = subprocess.run(["sumo", "--version"], capture_output=True, text=True)
    print("SUMO binary found. Version info:")
    print(result.stdout.split("\n")[0])
 
    # Confirm traci (the Python API for SUMO) imported successfully
    print(f"\nTraCI module loaded successfully: {traci.__name__}")
 
    print("\nEnvironment is ready for NeuroFlow-FL training.")
 
if __name__ == "__main__":
    main()