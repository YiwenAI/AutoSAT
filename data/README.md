1. Our data can be found in https://drive.google.com/drive/folders/1-au8hBbx4YAdJDlct9glCODpL0TQcYnA?usp=drive_link
 
2. We publish our method to obtaining all of our data.
* For CNP(Chromatic Number of the Plane, SAT Competition 2018,), we use examples from https://mat.tepper.cmu.edu/COLOR/instances.html#XXDSJ , with our [transform code](./CNP_generate/)

* For SCPC(Set Covering Problem with Conflict, SAT Competition 2022/2023), we only use the examples in SAT Competition

* For PRP(Profitable Robust Production, SAT Competition 2023), we use [`Picat`](http://picat-lang.org/) to generate cnf data keeping the question structure. (the same way from the Data Author)

* For CoinsGrid/KnightTour/LangFord/Zamkeller, are also generted by [`Picat`](http://picat-lang.org/)
