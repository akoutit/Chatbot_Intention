# Installer les dépendances nécessaires
def install_dependencies():
    import subprocess
    subprocess.call(["pip", "install", "torch", "pandas", "numpy", "transformers", "scikit-learn", "joblib", "nltk", "sentencepiece"])
    

install_dependencies()