# CNN Hyperparameter Optimization

This project investigates how data augmentation and hyperparameter settings influence the performance of convolutional neural networks on the Oxford-IIIT Pet Dataset.

The dataset contains images of 37 cat and dog breeds.

## Project Structure

```text
src/
    data.py
    model.py
    train.py
    experiments.py
    evaluate_final.py
    main.py

reports/
    results.md

requirements.txt
README.md



python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

base model run: python src/main.py

hyperparameter experiment run: python src/experiments.py

final evaluation run: python src/evaluate_final.py