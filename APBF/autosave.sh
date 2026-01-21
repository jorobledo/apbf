find . -name "*.md" -o -name "*.ipynb" -o -name "*.yml" | entr jupyter-book build .
