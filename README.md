# Frankls_RD9

Método entrópico ponderado RD₉ para la Conjetura de Frankl.  
Preimpresión LaTeX, figuras y código Python reproducible.  

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16884633.svg)](https://doi.org/10.5281/zenodo.16884633)

---

## 📑 Cita

Si utiliza este repositorio, cítelo de la siguiente manera:

**BibTeX**
```bibtex
@misc{FranklRD9,
  title   = {An RD9--Weighted Entropy Method for Improving Frankl's Union--Closed Sets Bound},
  author  = {Mauro González Romero and Tamara García Carnicero},
  year    = {2025},
  doi     = {10.5281/zenodo.16884633},
  url     = {https://doi.org/10.5281/zenodo.16884633},
  note    = {GitHub repository and Zenodo archived release}
}

``` 

## ⚙️ Cómo reproducir

Ejecuta el script para calcular y graficar los resultados:

```bash
python rd9_entropy_window.py --theta 0.33 --gamma 0.16 --beta 0.34 --grid --outdir .

``` 
Esto generará:

Delta_shape.png: curva de Δ(p) con p_max.

alpha_max_window_vs_beta.png: gráfico de α_max vs β.

Archivos CSV con valores numéricos.

APA
González Romero, M., & García Carnicero, T. (2025). Un método de entropía ponderada RD9 para mejorar la Conjetura de Frankl: conjuntos cerrados por unión. (Versión v1.0.3). Zenodo. https://doi.org/10.5281/zenodo.16884633

IEEE
M. González Romero y T. García Carnicero, Un método de entropía ponderada RD9 para mejorar la Conjetura de Frankl: conjuntos cerrados por unión, Zenodo, Versión v1.0.3, 2025. doi: 10.5281/zenodo.16884633.

📜 Licencia y derechos de autor

© 2025 Mauro González Romero y Tamara García Carnicero.

Código: Licenciado bajo la Licencia MIT.
Puede usar, modificar y distribuir libremente el código bajo las condiciones del MIT.

Artículo (LaTeX, PDF) y Figuras: Licenciados bajo CC BY-NC-ND 4.0 International.
Puede compartir esta obra con atribución, pero no se permite su uso comercial ni sus derivados.
