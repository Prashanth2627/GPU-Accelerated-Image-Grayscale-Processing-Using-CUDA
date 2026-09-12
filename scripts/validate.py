import os
import sys

def parse_ppm_pgm_header(filename):
    with open(filename, 'rb') as f:
        magic = f.readline().decode('ascii').strip()
        
        # Skip comments
        line = f.readline().decode('ascii')
        while line.startswith('#'):
            line = f.readline().decode('ascii')
            
        dims = line.strip().split()
        if len(dims) == 2:
            w, h = int(dims[0]), int(dims[1])
        else:
            w = int(dims[0])
            h = int(f.readline().decode('ascii').strip())
            
        return magic, w, h

def main():
    print("========================================")
    print("Project Validation Script")
    print("========================================")
    
    errors = 0
    
    # 1. Check directories
    required_dirs = ['src', 'input', 'output', 'results', 'scripts']
    for d in required_dirs:
        if os.path.isdir(d):
            print(f"[OK] Directory exists: '{d}/'")
        else:
            print(f"[FAIL] Directory missing: '{d}/'")
            errors += 1
            
    # 2. Check core project files
    required_files = [
        'src/main.cu',
        'Makefile',
        'README.md',
        'PRESENTATION.md',
        '.gitignore',
        'scripts/generate_data.py',
        'scripts/run.sh'
    ]
    for f in required_files:
        if os.path.isfile(f):
            print(f"[OK] File exists: '{f}'")
        else:
            print(f"[FAIL] File missing: '{f}'")
            errors += 1
            
    # 3. Check input PPM images
    ppm_images = ['image1.ppm', 'image2.ppm', 'image3.ppm', 'image4.ppm']
    for img in ppm_images:
        path = os.path.join('input', img)
        if os.path.isfile(path):
            try:
                magic, w, h = parse_ppm_pgm_header(path)
                if magic == 'P6':
                    print(f"[OK] Valid PPM Input: '{path}' ({w}x{h})")
                else:
                    print(f"[FAIL] Invalid PPM format in '{path}': Expected 'P6', got '{magic}'")
                    errors += 1
            except Exception as e:
                print(f"[FAIL] Could not parse PPM header for '{path}': {e}")
                errors += 1
        else:
            print(f"[FAIL] Missing PPM Input image: '{path}'")
            errors += 1
            
    # 4. Check output PGM images and verify dimensions match input
    for img in ppm_images:
        pgm_name = img.replace('.ppm', '.pgm')
        ppm_path = os.path.join('input', img)
        pgm_path = os.path.join('output', pgm_name)
        
        if os.path.isfile(pgm_path):
            try:
                pgm_magic, pgm_w, pgm_h = parse_ppm_pgm_header(pgm_path)
                
                # Check format
                if pgm_magic != 'P5':
                    print(f"[FAIL] Invalid PGM format in '{pgm_path}': Expected 'P5', got '{pgm_magic}'")
                    errors += 1
                    continue
                
                # Compare dimensions
                if os.path.isfile(ppm_path):
                    _, ppm_w, ppm_h = parse_ppm_pgm_header(ppm_path)
                    if pgm_w == ppm_w and pgm_h == ppm_h:
                        print(f"[OK] Valid PGM Output (dimensions match input): '{pgm_path}' ({pgm_w}x{pgm_h})")
                    else:
                        print(f"[FAIL] Dimension mismatch for '{pgm_name}': Input {ppm_w}x{ppm_h} vs Output {pgm_w}x{pgm_h}")
                        errors += 1
                else:
                    print(f"[OK] Valid PGM Output: '{pgm_path}' ({pgm_w}x{pgm_h})")
            except Exception as e:
                print(f"[FAIL] Could not parse PGM header for '{pgm_path}': {e}")
                errors += 1
        else:
            print(f"[FAIL] Missing PGM Output image: '{pgm_path}'")
            errors += 1
            
    # 5. Check logs and performance csv
    log_files = [
        ('results/execution_log.txt', 'Execution Log'),
        ('results/performance.csv', 'Performance Metrics')
    ]
    for path, desc in log_files:
        if os.path.isfile(path):
            size = os.path.getsize(path)
            if size > 0:
                print(f"[OK] {desc} exists and is non-empty: '{path}' ({size} bytes)")
            else:
                print(f"[FAIL] {desc} is empty: '{path}'")
                errors += 1
        else:
            print(f"[FAIL] Missing {desc}: '{path}'")
            errors += 1
            
    print("========================================")
    if errors == 0:
        print("[SUCCESS] All validation checks PASSED!")
        print("Your project is ready to compile and run in a CUDA environment.")
        print("The repository matches the required structure and contains all proof files.")
        print("========================================")
        sys.exit(0)
    else:
        print(f"[ERROR] Validation failed with {errors} error(s).")
        print("Please correct the issues and try again.")
        print("========================================")
        sys.exit(1)

if __name__ == '__main__':
    main()
