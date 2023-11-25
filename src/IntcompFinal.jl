
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

data_train = data[1:70, :]
data_test = data[71:100, :]

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