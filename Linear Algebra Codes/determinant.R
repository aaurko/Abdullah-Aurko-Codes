Det1 <- function(Det){
  a <- Det[1,1]
  b <- Det[1,2]
  c <- Det[2,1]
  d <- Det[2,2]
  return (a*d - b*c)} 

M = matrix(1:4,2,2) 

#command
Det1(M)