#define function to calculate cross product 
crossproduct <- function(x, y, i=1:3) {
  create3D <- function(x) head(c(x, rep(0, 3)), 3)
  x <- create3D(x)
  y <- create3D(y)
  j <- function(i) (i-1) %% 3+1
  return (x[j(i+1)]*y[j(i+2)] - x[j(i+2)]*y[j(i+1)])
}

#define vectors
v1 <- c(3, 1, -5)
v2 <- c(-8, 0, 9)

#calculate cross product
crossproduct(v1, v2)