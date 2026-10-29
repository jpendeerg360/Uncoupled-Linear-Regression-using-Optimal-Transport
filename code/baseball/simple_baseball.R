#!/usr/bin/env Rscript
# Run: Rscript --vanilla simple_baseball.R
# Base R only. Data are embedded; writes exactly two PNGs in output/.
# One-dimensional simplification of Jaffe, Ignatiadis & Sen's baseball example.
# Source: https://github.com/aqjaffe/constrained-denoising-EB-OT
# Pinned data commit: 71f03d26301f84bda307c27fdc3329b9a9a814ca
# 308 first-observed MLB-season players: 120 (2022), 82 (2023), 106 (2024).
# Selected from ID-linked previous-year minor-league records; pitchers removed.
# Earlier MLB records are checked only within available 2021-2024 files.
# RBI and games below are from the corresponding major-year batting CSV.

f <- grep("^--file=",commandArgs(FALSE),value=TRUE)
ROOT <- if(length(f)) dirname(normalizePath(sub("^--file=","",f[1]))) else getwd()
OUT <- file.path(ROOT,"output")
player_id <- c(
  23697, 26288, 25931, 24679, 25764, 14366, 21897, 27676, 20970, 20538, 21707, 26294, 22244, 25976, 21270, 24782, 26031, 26289, 25055, 23792,
  22471, 18821, 22823, 19296, 19813, 19983, 19273, 20505, 25961, 22277, 24770, 25660, 22514, 25111, 23818, 20543, 27684, 20529, 23359, 20926,
  23733, 22261, 25867, 24273, 26121, 26151, 22715, 15697, 22505, 19859, 20122, 16680, 18400, 22263, 25816, 24729, 13408, 19779, 17910, 19964,
  12394, 19563, 20533, 18603, 20510, 22254, 26323, 23794, 17473, 26123, 18690, 26396, 20805, 24703, 25353, 18587, 22093, 23003, 21634, 17737,
  20022, 20450, 22117, 25845, 22563, 21837, 24064, 20165, 22184, 27615, 22411, 13750, 20017, 19768, 20599, 20002, 22906, 19938, 21865, 25459,
  22650, 26368, 7226, 17040, 21780, 22896, 24589, 20178, 21853, 25543, 22202, 22557, 22515, 22169, 20054, 19625, 22197, 27465, 19458, 21386,
  29695, 27534, 29766, 23312, 23565, 26668, 27647, 25807, 24336, 27790, 29622, 27690, 27475, 27815, 23800, 25505, 29931, 30011, 23772, 26517,
  26149, 25332, 25479, 19455, 16496, 21716, 25805, 20311, 28080, 27630, 27562, 26365, 23698, 19877, 26208, 25813, 19358, 26477, 24262, 19788,
  22842, 20626, 18123, 23296, 24597, 22232, 24388, 22054, 25493, 12140, 21587, 25822, 23985, 22186, 23691, 27531, 29976, 19212, 27769, 29715,
  24451, 27604, 19957, 26420, 26467, 26438, 21497, 23690, 25569, 21547, 29591, 26319, 10441, 18217, 25482, 27577, 27845, 23695, 27479, 23788,
  30162, 27467, 29490, 28806, 33333, 27899, 24816, 24488, 29518, 22857, 27493, 26548, 29554, 29634, 24257, 23782, 25782, 33541, 21626, 29547,
  29575, 25979, 22766, 25425, 29573, 25180, 25483, 25705, 29912, 25575, 25448, 24605, 23372, 29592, 21797, 31437, 26599, 33266, 29630, 26295,
  31394, 25475, 29794, 23395, 26540, 27471, 24496, 25524, 21496, 31781, 19722, 27883, 29487, 27464, 23968, 14412, 23250, 23557, 27698, 25967,
  23789, 27504, 22707, 29949, 27599, 21682, 27543, 23442, 25971, 21730, 22045, 19910, 27784, 27462, 31562, 25690, 29782, 15504, 31396, 28253,
  22530, 19384, 25477, 25138, 26374, 29576, 24598, 27789, 31595, 19292, 29788, 31363, 26469, 20557, 24759, 29524, 25621, 27763, 27962, 29606,
  26224, 31349, 31765, 27781, 28312, 31567, 27501, 31431)
rbi <- c(
  75, 42, 64, 45, 80, 34, 47, 26, 43, 66, 19, 49, 38, 42, 11, 24, 18, 18, 12, 39,
  9, 10, 2, 13, 11, 11, 10, 18, 10, 19, 3, 8, 12, 5, 6, 20, 17, 13, 26, 20,
  25, 9, 6, 0, 1, 11, 2, 0, 12, 10, 6, 0, 30, 35, 22, 2, 3, 2, 0, 0,
  0, 0, 0, 0, 0, 0, 8, 2, 2, 5, 6, 16, 2, 6, 0, 0, 1, 1, 6, 0,
  1, 0, 20, 12, 5, 6, 2, 5, 3, 3, 1, 3, 0, 0, 1, 2, 0, 0, 0, 11,
  1, 16, 3, 5, 2, 1, 10, 8, 3, 4, 4, 0, 0, 17, 7, 6, 62, 28, 17, 4,
  50, 37, 32, 54, 20, 44, 60, 21, 23, 12, 24, 42, 51, 23, 13, 11, 12, 37, 14, 15,
  14, 13, 48, 27, 2, 19, 44, 33, 7, 10, 13, 6, 9, 1, 5, 1, 2, 7, 5, 18,
  6, 3, 2, 2, 0, 0, 0, 9, 4, 0, 0, 4, 12, 34, 3, 0, 0, 0, 1, 17,
  4, 6, 2, 1, 14, 21, 10, 8, 16, 3, 4, 7, 12, 6, 1, 30, 11, 10, 12, 1,
  19, 7, 90, 79, 74, 61, 46, 34, 41, 37, 11, 6, 19, 11, 28, 19, 27, 8, 12, 9,
  17, 1, 1, 3, 8, 22, 8, 3, 0, 15, 1, 0, 1, 20, 15, 3, 5, 3, 25, 7,
  8, 5, 15, 10, 11, 5, 1, 7, 1, 23, 0, 4, 1, 11, 1, 0, 0, 0, 0, 0,
  21, 5, 6, 2, 4, 1, 0, 1, 0, 0, 0, 0, 0, 16, 8, 10, 8, 7, 3, 2,
  5, 1, 0, 1, 0, 23, 28, 5, 27, 2, 12, 5, 2, 0, 0, 0, 11, 6, 0, 21,
  12, 10, 0, 0, 0, 10, 1, 13)
games <- c(
  132, 113, 114, 126, 150, 56, 113, 72, 91, 125, 44, 127, 106, 93, 27, 52, 41, 34, 44, 113,
  18, 98, 18, 47, 48, 22, 27, 32, 31, 100, 4, 11, 27, 16, 10, 53, 52, 28, 77, 39,
  58, 55, 26, 1, 5, 34, 9, 8, 30, 25, 28, 2, 76, 89, 40, 5, 12, 5, 3, 7,
  3, 1, 4, 3, 1, 1, 28, 17, 4, 11, 14, 73, 7, 58, 25, 3, 5, 6, 7, 2,
  2, 6, 49, 31, 16, 32, 9, 32, 16, 16, 23, 14, 8, 5, 11, 11, 4, 5, 5, 32,
  15, 65, 13, 47, 17, 11, 54, 18, 28, 12, 14, 9, 9, 69, 39, 22, 129, 110, 66, 39,
  89, 109, 69, 128, 35, 98, 159, 54, 59, 23, 57, 132, 117, 68, 37, 34, 33, 63, 28, 35,
  28, 57, 126, 92, 15, 49, 110, 93, 8, 19, 19, 24, 26, 2, 10, 26, 8, 15, 28, 54,
  6, 11, 29, 4, 8, 6, 1, 22, 15, 9, 13, 13, 31, 137, 18, 15, 14, 5, 13, 23,
  17, 46, 16, 15, 76, 59, 33, 23, 51, 18, 26, 27, 36, 33, 25, 90, 27, 27, 37, 17,
  75, 45, 156, 148, 134, 148, 116, 133, 79, 112, 24, 14, 25, 24, 67, 63, 68, 31, 30, 33,
  36, 51, 28, 19, 16, 73, 31, 2, 9, 54, 6, 9, 5, 45, 33, 34, 26, 28, 81, 41,
  33, 15, 53, 18, 43, 22, 6, 39, 12, 60, 6, 18, 19, 27, 8, 1, 1, 1, 1, 1,
  61, 11, 13, 18, 29, 13, 11, 11, 3, 5, 3, 3, 3, 93, 19, 37, 30, 21, 16, 15,
  13, 12, 11, 7, 8, 50, 69, 20, 50, 16, 36, 13, 27, 4, 13, 5, 32, 16, 10, 71,
  48, 53, 8, 7, 17, 61, 15, 55)
stopifnot(length(rbi)==308,length(games)==308,length(unique(player_id))==308,
          all(rbi>=0),all(games>0))

# Gamma(shape=a, rate=b) prior; Poisson(games * theta) likelihood.
# Integrating theta gives a negative-binomial marginal likelihood.
fit_prior <- function() {
  nll <- function(q) {
    a <- exp(q[1]);b <- exp(q[2])
    -sum(dnbinom(rbi,size=a,mu=games*a/b,log=TRUE))
  }
  fits <- lapply(list(c(1,3),c(2,6),c(5,15)),function(s)
    optim(log(s),nll,method="L-BFGS-B",lower=rep(log(1e-4),2),
          upper=rep(log(1e4),2),control=list(factr=1e5,maxit=2000)))
  values <- vapply(fits,function(x)x$value,numeric(1))
  stopifnot(all(vapply(fits,function(x)x$convergence==0,logical(1))),
            max(values)-min(values)<1e-6)
  q <- fits[[which.min(values)]]$par
  stopifnot(all(q>log(1e-4)+1e-4),all(q<log(1e4)-1e-4))
  exp(q)
}
v <- function(x) mean((x-mean(x))^2) # empirical variance: denominator N
main <- function() {
  ab <- fit_prior();a <- ab[1];b <- ab[2]
  mu <- a/b;A <- a/b^2
  raw <- rbi/games
  eb <- (rbi+a)/(games+b)
  bw <- mu+sqrt(A/v(raw))*(raw-mean(raw))
  vc <- mu+sqrt(A/v(eb))*(eb-mean(eb))
  pred <- cbind("Observed rate"=raw,"Direct BW"=bw,"EB"=eb,"Variance-constrained EB"=vc)
  post_var <- (rbi+a)/(games+b)^2
  # Conditional risk under the fitted model, not an observed latent MSE.
  risk <- colMeans((pred-eb)^2+post_var)
  vv <- apply(pred,2,v);mm <- colMeans(pred)
  stopifnot(max(abs(vv[c(2,4)]-A))<1e-12,max(abs(mm[c(2,4)]-mu))<1e-12,
            risk[3]<=risk[4]+1e-12,risk[4]<=risk[2]+1e-12,
            abs(risk[4]-risk[3]-(mu-mean(eb))^2-(sqrt(A)-sqrt(v(eb)))^2)<1e-12)
  cat(sprintf("Gamma shape=%.6f, rate=%.6f; latent mean=%.6f, variance=%.6f\n",a,b,mu,A))
  print(data.frame(method=colnames(pred),mean=mm,variance=vv,variance_ratio=vv/A,
                   posterior_MSE=risk,negative=colSums(pred<0),row.names=NULL),digits=6,row.names=FALSE)
  cat("Checks passed: optimization and both moment constraints.\n")
  # Do not clip negative estimates: clipping would change the constraints.
  dir.create(OUT,showWarnings=FALSE,recursive=TRUE)
  cols <- c("#7d8791","#315d8a","#da8b26","#18836f")
  ord <- order(raw,games);lim <- range(c(0,pred))
  png(file.path(OUT,"baseball_estimates.png"),width=1800,height=1250,res=180)
  par(mfrow=c(2,2),mar=c(4.5,4.1,2.6,.9),family="sans")
  for(k in 1:4) {
    plot(seq_along(ord),pred[ord,k],pch=19,cex=.55,col=cols[k],ylim=lim,
         xlab="Players ordered by observed RBI per game",ylab="RBI per game",
         main=colnames(pred)[k],cex.main=1)
    abline(h=mu,lty=2,col="#444444")
  }
  dev.off()
  png(file.path(OUT,"baseball_variance_risk.png"),width=1800,height=850,res=180)
  par(mfrow=c(1,2),mar=c(4.7,4.5,3.3,1.1),family="sans")
  labels <- c("Observed","BW","EB","VC-EB")
  x <- barplot(vv/A,names.arg=labels,col=cols,border=NA,ylim=c(0,max(vv/A)*1.2),
       ylab="Empirical variance / fitted latent variance",main="Between-player variation",cex.names=.9)
  abline(h=1,lty=2,col="#444444");text(x,vv/A,labels=sprintf("%.2f",vv/A),pos=3,cex=.85)
  x <- barplot(risk*1e3,names.arg=labels,col=cols,border=NA,ylim=c(0,max(risk*1e3)*1.2),
       ylab=expression("Posterior MSE " %*% 10^3),main="Risk under the fitted model",cex.names=.9)
  text(x,risk*1e3,labels=sprintf("%.2f",risk*1e3),pos=3,cex=.85)
  mtext("Model-based risk; true player rates are unobserved",side=1,line=3.5,cex=.75)
  dev.off()
}
if(sys.nframe()==0) main()
