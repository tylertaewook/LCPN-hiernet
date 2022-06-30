for file in ../../split_dataset/*; do
    echo "=======TRAINING: $(basename "$file")========"
    python3 main.py -d $(basename "$file")
done