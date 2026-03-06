#!/bin/bash
g++ -std=c++17 main.cpp numerics.cpp -O2 -fopenmp -o main && ./main "$@"
python3 render.py --out recent.gif --cmap Blues_r

if command -v explorer.exe >/dev/null 2>&1 && command -v wslpath >/dev/null 2>&1; then
	explorer.exe "$(wslpath -w ./sims/recent.gif)"
elif command -v xdg-open >/dev/null 2>&1; then
	xdg-open ./sims/recent.gif >/dev/null 2>&1 || true
fi