# Recognition of handwritten - Using HuMoments with SVM

From my tests it seems difficult to characterize a certain pattern of accuracy results depending
on 𝜸 or c where each time one of them is a constant (In general, it is difficult to extract information
by looking at csv files) and the other is variable. To get a slightly clearer picture, I attach graphs
of the accuracy function, as a function of 𝜸 and c, such that –

https://user-images.githubusercontent.com/32679759/58994529-7faa8980-87f9-11e9-9470-0105c704e5ad.png

As we can see, 𝜸 is constant in the kernel function and C used to set the amount of regularization
for the error function in which the scikit-learn library uses – " squared hinge loss".

In the following graph we can see the accuracy results of the algorithm for different 𝛾 and c values, where
the accuracy function result is the average of the correct prediction. SVM training is done by vectors
obtained from calculating the HuMoments of images from MNIST data, after the log transform has been performed.

https://user-images.githubusercontent.com/32679759/58994583-b84a6300-87f9-11e9-9a95-840a74ac479a.png

After running the program for higher resolution of c and gamma values in smaller domain The pattern became sharper and clearer.

https://user-images.githubusercontent.com/32679759/58994632-e334b700-87f9-11e9-96fb-6817d8eecc5a.png

I attach some results of specific simulations in the form of confusion matrix to better understand the
algorithm error between the different classes:

https://user-images.githubusercontent.com/32679759/58994712-2858e900-87fa-11e9-8c33-662fcbcac673.png

It is possible that the seven HuMoments alone are not enough to describe the Images.
Probably by adding more features, not with HuMoment alone, accuracy can be improved – https://pdfs.semanticscholar.org/d9d1/5738be258591fb19d3373a1d9435eaab4998.pdf .
