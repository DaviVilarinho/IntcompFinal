
#module IntcompFinal
using Flux
using CSV, DataFrames
using DataFrames
using Dates
using Random
using Statistics
using CategoricalArrays
using DataFrames
using JLD2

function cardinalToRadians(cardinal_point::Union{AbstractString,Missing,String})::Union{Float64,Missing}
  if ismissing(cardinal_point) || cardinal_point == "NA"
    return 0
  end
  cardinal_dict = Dict("N" => 0.0, "NNE" => π / 8, "NE" => π / 4, "ENE" => 3 * π / 8,
    "E" => π / 2, "ESE" => 5 * π / 8, "SE" => 3 * π / 4, "SSE" => 7 * π / 8,
    "S" => π, "SSW" => -7 * π / 8, "SW" => -3 * π / 4, "WSW" => -5 * π / 8,
    "W" => -π / 2, "WNW" => -3 * π / 8, "NW" => -π / 4, "NNW" => -π / 8)

  return get(cardinal_dict, cardinal_point, NaN)
end

function readAndProcessData(lines::Int64=0)
  column_types = Dict(
    :Date => String,
    :Location => String,
    :MinTemp => Float64,
    :MaxTemp => Float64,
    :Rainfall => Float64,
    :Evaporation => Float64,
    :Sunshine => Float64,
    :WindGustSpeed => Float64,
    :WindDir9am => String,
    :WindGustDir => String,
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
  dataframe = DataFrame(CSV.File("weatherAUS.csv", missingstring=["NA"], dateformat=Dates.ISODateFormat, types=column_types))

  if lines > 0
    dataframe = dataframe |> x -> first(x, lines)
  end
  for directionColumn in [:WindDir3pm, :WindDir9am, :WindGustDir]
    dataframe[!, directionColumn] .= cardinalToRadians.(dataframe[!, directionColumn])
  end
  # primeiro, dataframe sem os spikes...
  locs = unique(dataframe.Location)
  locDict = Dict()
  for i in eachindex(locs)
    locDict[locs[i]] = Float64(i)
  end

  dataframe.Location = map(l -> get(locDict, locDict, 0), dataframe.Location)
  dataframe.Date = map(d -> datetime2unix(DateTime(d)), dataframe.Date)
  dataframe.RainToday .= coalesce.(dataframe.RainToday, "No")
  dataframe.RainTomorrow .= coalesce.(dataframe.RainTomorrow, "No")
  dataframe.RainToday = dataframe.RainToday .== "Yes"
  dataframe.RainTomorrow = dataframe.RainTomorrow .== "Yes"

  dataframe.RainTomorrow = map(tomorrow -> tomorrow == 1 ? [false, true] : [true, false], dataframe.RainTomorrow)

  dataframe .= coalesce.(dataframe, 0.0)

  #for categoricalColumn in [:Location, :WindDir3pm, :WindDir9am, :WindGustDir]
  #  dataframe[!, categoricalColumn] = categorical(dataframe[!, categoricalColumn])
  #end

  return dataframe
end
inputdata = 1000
data = readAndProcessData(inputdata)

Flux.Random.seed!(42)

indices = shuffle(1:size(data, 1))
split_idx = Int(round(0.7 * size(data, 1)))

train_indices = indices[1:split_idx]
test_indices = indices[split_idx+1:end]

x_train = select(data, Not([:RainTomorrow]))[train_indices, :]
x_test = select(data, Not([:RainTomorrow]))[test_indices, :]
y_train = select(data, [:RainTomorrow])[train_indices, :]
y_test = select(data, [:RainTomorrow])[test_indices, :]

mlp_hidden = 10
model = Chain(
  Dense(size(x_train, 2) => mlp_hidden, tanh),
  Dense(mlp_hidden => mlp_hidden, relu),
  Dense(mlp_hidden => 2, σ),
  softmax)
optim = Flux.setup(Flux.Descent(0.15), model)

losses = []
epoch = 1_000
for epoch in 1:epoch#_000
  for i in 1:size(x_train, 1)
    x_t = collect(x_train[i, :])
    y_t = y_train.RainTomorrow[i]
    e_loss, grads = Flux.withgradient(model) do m
      y_hat = m(x_t)
      Flux.crossentropy(y_hat, y_t)
    end
    Flux.update!(optim, model, grads[1])
    push!(losses, e_loss)
  end
end

correct = []

for i in 1:size(x_test, 1)
  if ((model(collect(x_test[i, :]))[1] >= 0.5) == y_test.RainTomorrow[i][1])
    push!(correct, 1)
  end
end

n_correct = length(correct)
n_test = size(x_test, 1)
print("$n_correct / $n_test (", n_correct * 100 / n_test, "%) corretos")

model_state = Flux.state(model);

nowmoment = string(Dates.now())

jldsave("./models/$nowmoment-E$epoch-MLP$mlp_hidden-$inputdata-$n_correct-out-of-$n_test.jld2"; model_state)

#end