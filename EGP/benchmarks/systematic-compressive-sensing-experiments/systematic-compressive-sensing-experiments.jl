include("../run_experiments.jl")

settings = [
    (ids = 1, n = 100, p = 10, s1 = 5, s2 = 1000000), # low
    (ids = 2, n = 50, p = 1000, s1 = 5, s2 = 1000000), # high-5
    (ids = 3, n = 100, p = 1000, s1 = 10, s2 = 1000000), # high-10
    (ids = 4, n = 100, p = 10000, s1 = 10, s2 = 1000000), # very high-10
]

settings = settings[1:1]
rhos = Float32.([0, 0.35, 0.7, 0.9])
rhos = rhos[1:1]
SNRs = logspace(0.05,20,10)

run_experiments(settings; repetitions=2, rhos = rhos, SNRs = SNRs)