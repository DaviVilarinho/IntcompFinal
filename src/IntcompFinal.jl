
module IntcompFinal
using Flux
using CSV
using DataFrames
using Statistics: mean, std
using Dates
using CategoricalArrays

column_types = Dict(
  :Date => String,
  :Location => String,
  :MinTemp => Float64,
  :MaxTemp => Float64,
  :Rainfall => Float64,
  :Evaporation => Float64,
  :Sunshine => Float64,
  :WindGustDir => String,
  :WindGustSpeed => Float64,
  :WindDir9am => String,
  :WindDir3pm => String,
  :WindSpeed9am => Float64,
  :WindSpeed3pm => Float64,
  :Humidity9am => Float64,
  :Humidity3pm => Float64,
  :Pressure9am => Float64,
  :Pressure3pm => Float64,
  :Cloud9am => Float64,
  :Cloud3pm => Float64,
  :Temp9am => Float64,
  :Temp3pm => Float64,
  :RainToday => String,
  :RainTomorrow => String
)

data = CSV.File("weatherAUS.csv", types=column_types, missingstrings=["NA"]) |> DataFrame# |> x -> first(x, 10000)
data.RainToday .= data.RainToday .== "Yes"
data.RainTomorrow .= data.RainTomorrow .== "Yes"

#categorical_directions = CategoricalArray(["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"])
data = coalesce.(data, 0)

columns_to_delete = [:WindGustDir, :WindDir9am, :WindDir3pm, :Location]

select!(data, Not(columns_to_delete))

data[!, :Date] = Dates.datetime2unix.(DateTime.(data[!, :Date]))

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

model = Chain(
  Dense(size(X_train, 2), 64, relu),
  Dense(64, 1),
  softmax
)

loss(x, y) = Flux.mse(model(x), y)

optimizer = ADAM()

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
