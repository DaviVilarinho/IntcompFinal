
module IntcompFinal
using Flux
using Statistics: mean, std
include("RainManipulation.jl")
using .RainManipulation

data = readAndPreprocessData(100)
X = Matrix(data[:, setdiff(names(data), [:RainTomorrow])])
y = data.RainTomorrow

variable_names = names(data)
variable_types = eltype.(eachcol(data))

# Print variable names and types
for (name, typ) in zip(variable_names, variable_types)
  println("Variable Name: $name, Type: $typ")
end
split_ratio = 0.8
split_idx = Int(round(size(X, 1) * split_ratio))
X_train, X_test = X[1:split_idx, :], X[split_idx+1:end, :]
y_train, y_test = y[1:split_idx], y[split_idx+1:end]


println("Creating model")

model = Chain(
  Dense(size(X_train, 2), 64, relu),
  Dense(64, 1),
  softmax
)

println("Model Created")
loss(x, y) = Flux.mse(model(x), y)

optimizer = ADAM()

println("training")

data_train = [(X_train[i, :], y_train[i]) for i in 1:size(X_train, 1)]
Flux.train!(loss, Flux.params(model), data_train, optimizer)

print(size(X_test))
y_pred = []

for i in 1:size(X_test, 1)
  push!(y_pred, model(X_test[i, :]))
end

y_pred = hcat(y_pred...)[:]

mse = Flux.mse(y_pred, y_test)
println("Mean Squared Error on Test Set: $mse")
end
