
#module IntcompFinal
include("RainManipulation.jl")
using .RainManipulation
using Flux
using DataFrames

data = readAndPreprocessData(100)
data.DeltaTemp = data.MaxTemp .- data.MinTemp
data.RainTomorrow = map(tomorrow -> tomorrow == 1 ? [false, true] : [true, false], data.RainTomorrow)

data = select(data, [:DeltaTemp, :RainTomorrow])
data = map(row -> (row.DeltaTemp, row.RainTomorrow), eachrow(data))

Flux.Random.seed!(42)

indices = shuffle(1:size(data, 1))
split_idx = Int(round(0.7 * size(data, 1)))

train_indices = indices[1:split_idx]
test_indices = indices[split_idx+1:end]

data_train = data[train_indices, :]
data_test = data[test_indices, :]

model = Chain(
  Dense(1 => 2, σ),
  Dense(2 => 2),
  softmax)
optim = Flux.setup(Flux.Adam(0.01), model)

losses = []
for epoch in 1:10#_000
  for (x_t, y_t) in data_train
    e_loss, grads = Flux.withgradient(model) do m
      y_hat = m([x_t])
      Flux.crossentropy(y_hat, y_t)
    end
    Flux.update!(optim, model, grads[1])
    push!(losses, e_loss)
  end
end

correct = []

for (Δ, hotmax) in data_test
  if ((model([Δ])[1] >= 0.5) == hotmax[1])
    push!(correct, 1)
  end
end

print(length(correct), "/ 30 corretos")

#end