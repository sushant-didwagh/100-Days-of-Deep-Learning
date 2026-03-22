Normalization 
Why need: in dataset any column/feature is dominating it put more impact on accuracy lead to unbais traing accuracy also lead to overfitting
1. standardization -> no idea about data values ex- salary (0 - not define)
   when normal distribution
   formula = Xi - mean / std(dev)
   it scale data to -1 to 1 range
2. normalization -> when min and max values known eg. CGPA 0 to 10.
   when outliers present
   formula = Xi - Xmin / Xmax - Xmin
   it scale data to 0 to 1 in range.
   
