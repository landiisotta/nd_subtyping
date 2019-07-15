# Title     : TODO
# Objective : TODO
# Created by: ilandi
# Created on: 29/05/2019

test_demog <- function(commandArgs(TRUE)[1],
                       commandArgs(TRUE)[2]){

}

level.path <- 'level-4'
data.path <- file.path('~/Documents/nd_subtyping/data/odf-data-2019-06-13-12-28-37/', level.path)

cl_df = read.table(file.path(data.path, 'cluster-stats.csv'), sep=',',
                   header=TRUE,
                   as.is = TRUE)

cl_df$CLUSTER <- as.factor(cl_df$CLUSTER)


pairwise.t.test(cl_df$AGE, cl_df$CLUSTER, p.adjust.method='bonferroni')

pairwise.t.test(cl_df$N_ENCOUNTERS, cl_df$CLUSTER, p.adjust.method='bonferroni')

# tab <- table(cl_df$SEX, cl_df$CLUSTER)

pairwise.chisq.test <- function(x, g, p.adjust.method = p.adjust.methods, ...) {
  DNAME <- paste(deparse(substitute(x)), "and", deparse(substitute(g)))
  g <- factor(g)
  p.adjust.method <- match.arg(p.adjust.method)

  compare.levels <- function(i, j) {
    idx <- which(as.integer(g) == i | as.integer(g) == j)
    xij <- x[idx]
    gij <- as.character(g[idx])
    gij <- as.factor(gij)
    print(table(xij, gij))
    chisq.test(xij, gij, ...)$p.value
  }
  PVAL <- pairwise.table(compare.levels, levels(g), p.adjust.method)
  ans <- list(method = "chi-squared test", data.name = DNAME, p.value = PVAL,
              p.adjust.method = p.adjust.method)
  class(ans) <- "pairwise.htest"
  ans
}

pairwise.chisq.test(cl_df$SEX, cl_df$CLUSTER, p.adjust.method='bonferroni')