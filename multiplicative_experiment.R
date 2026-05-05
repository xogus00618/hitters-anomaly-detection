library(ISLR)
library(isotree)
set.seed(123)

data(Hitters)
hitters_raw <- na.omit(Hitters)
data_clean <- hitters_raw[!rownames(hitters_raw) == "-Pete Rose",]
data_numeric <- data_clean[, sapply(data_clean, is.numeric)]
hitters <- data_numeric[, colnames(data_numeric) != "Salary"]

col_means <- colMeans(hitters)
col_means[c("CAtBat", "CHits")] <- c(8000, 1000)

sample_sizes <- c(30, 50, 100)
ntrees <- 100

for (mult in 1:5) {
  base <- hitters
  if (mult > 1) {
    for (k in 1:(mult-1)) {
      noise <- matrix(rnorm(nrow(hitters) * ncol(hitters), 0, 1),
                      nrow = nrow(hitters), ncol = ncol(hitters))
      colnames(noise) <- colnames(hitters)
      base <- rbind(base, hitters + noise)
    }
  }
  
  synth <- rbind(base, col_means)
  n_out <- nrow(synth)
  
  pca_full <- prcomp(synth, scale = TRUE)
  ax_full <- predict(pca_full, newdata = synth)
  
  cat("\n", mult, "배 (관측값", nrow(base), "개) \n")
  
  for (s in sample_sizes) {
    # 1. iForest
    t1 <- system.time({
      iso1 <- isolation.forest(synth, sample_size = s, ntrees = ntrees, seed = 123)
      scores1 <- predict(iso1, synth)
    })
    r1 <- which(order(scores1, decreasing = TRUE) == n_out)
    
    # 2. 전체 PCA + iForest
    t2 <- system.time({
      iso2 <- isolation.forest(ax_full, sample_size = s, ntrees = ntrees, seed = 123)
      scores2 <- predict(iso2, ax_full)
    })
    r2 <- which(order(scores2, decreasing = TRUE) == n_out)
    
    # 3. 샘플 PCA + iForest
    t3 <- system.time({
      score_sum <- numeric(n_out)
      for (i in 1:ntrees) {
        idx <- sample(1:n_out, s)
        spl <- synth[idx,]
        pca <- prcomp(spl, scale = TRUE)
        ax <- predict(pca, newdata = synth)
        iso <- isolation.forest(ax, sample_size = s, ntrees = 1, seed = 123)
        score_sum <- score_sum + predict(iso, ax)
      }
      mean_score <- score_sum / ntrees
    })
    r3 <- which(order(mean_score, decreasing = TRUE) == n_out)
    
    # 4. EiF ndim=2, 3, 16
    for (nd in c(2, 3, ncol(synth))) {
      t4 <- system.time({
        iso4 <- isolation.forest(synth, sample_size = s, ntrees = ntrees,
                                 ndim = nd, seed = 123)
        scores4 <- predict(iso4, synth)
      })
      r4 <- which(order(scores4, decreasing = TRUE) == n_out)
      cat("EiF ndim=", nd, "rank:", r4, "time:", t4["elapsed"], "| ")
    }
    cat("\n")
    
    cat("ss=", s, 
        "| iF:", r1, "/", t1["elapsed"], "s",
        "/ 전체PCA:", r2, "/", t2["elapsed"], "s",
        "/ 샘플PCA:", r3, "/", t3["elapsed"], "s\n")
  }
}
