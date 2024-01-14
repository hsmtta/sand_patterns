# Sand Patterns
Simulate sand patterns driven by wind, based on Werner model. The time evolution is saved as a video.

# Samples
See `--help` option for more detail.

## Wind Ripple
```bash
python simulate.py  # Simulates with default settings
python simulate.py -w 6.0  # Sets wind speed to 6.0
```

[![Sand pattern (wind speed = 3.0) - YouTube](http://img.youtube.com/vi/aLJXOW_nKhM/0.jpg)](http://www.youtube.com/watch?v=aLJXOW_nKhM "Sand pattern (wind speed = 3.0) - YouTube")

## Dune
```bash
python simulate.py -d  # Enables dune simulation mode, forming Barchan dunes by default
python simulate.py -d -q 3.0  # Sets sand quantity to 3.0, forming transverse ridges

# Sets sand quantity, num iteration, grid size, and random seed
python simulate.py -d -q 3.0 -i 5000 -g 140 -s 0
```

[![Dune pattern (sand quantity = 1.0) - YouTube](http://img.youtube.com/vi/oiSAykjHsM0/0.jpg)](http://www.youtube.com/watch?v=oiSAykjHsM0 "Dune pattern (sand quantity = 1.0) - YouTube")

[![Dune pattern (sand quantity = 3.0) - YouTube](http://img.youtube.com/vi/eT4fVW0XDcY/0.jpg)](http://www.youtube.com/watch?v=eT4fVW0XDcY "Dune pattern (sand quantity = 3.0) - YouTube")

# Reference
1. [H. Nishimori, Geomorphological dynamics of aeolian sand: wind ripples and dunes, 1993 (in Japanese)](https://core.ac.uk/download/pdf/39226746.pdf)
2. [J. Elder, Dune field simulator](https://smallpond.ca/jim/sand/dunefieldMorphology/index.html)
