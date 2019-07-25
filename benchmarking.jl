using Main.LDLFactorizations, BenchmarkTools, Test, SparseArrays, LinearAlgebra, MatrixMarket, DelimitedFiles

open("benchmark.log", "w") do logfile
      println(logfile, "Problem & tref & mem_ref & t/tref & mem/mem_ref & tcopy/tref & memcopy/mem_ref \\")

problems = ["aug2d";
            "dual1";
            "genhs28";
            "hs21";
            "liswet1";
            "mosarqp1";
            "primal1";
            "s268";
            "zecevic2";
           ]

  for p in problems

    # read
    Afull = MatrixMarket.mmread("sqd-collection/$p/3x3/iter_0/K_0.mtx")
    rhs = readdlm("sqd-collection/$p/3x3/iter_0/rhs_0.rhs")[:]
    Aonlyupper = sparse(UpperTriangular(Afull))

    LDLref = ldl(Afull)
    LDLonlyupper = ldl(Aonlyupper, onlyupper = true)

    # tests
    @test norm(Afull * (LDLref \ rhs) - rhs) < 1e-6
    @test norm(Afull * (LDLonlyupper \ rhs) - rhs) < 1e-6

    # benchmarks

    benchmark_ref = @benchmark ldl($Afull)                                 samples=10 evals=1
    benchmark_onlyupper = @benchmark ldl($Aonlyupper, onlyupper = true)    samples=10 evals=1
    benchmark_getfull = @benchmark sparse(Symmetric($Aonlyupper, :U))       samples=10 evals=1

    tref = round(benchmark_ref.times[1]/1e9, digits = 6)
    memref = round(benchmark_ref.memory[1]/1e6, digits = 6)
    trel = round((benchmark_onlyupper.times[1]/1e9) / tref, digits = 3)
    memrel = round((benchmark_onlyupper.memory[1]/1e6) / memref, digits = 3)
    tgetfull = round((benchmark_getfull.times[1]/1e9) / tref, digits = 3)
    memgetfull = round((benchmark_getfull.memory[1]/1e6) / memref, digits = 3)

    println(logfile, " $p & $(tref) s & $(memref) MiB & $(trel) & $(memrel) & $(tgetfull) & $(memgetfull) \\ ")
  end
end
