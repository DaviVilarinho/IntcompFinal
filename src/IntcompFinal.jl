
#module IntcompFinal
using Flux
using DelimitedFiles, DataFrames
using DataFrames
using Dates
using CategoricalArrays
using DataFrames

function cardinalToRadians(cardinal_point::Union{AbstractString,Missing})::Union{Float64,Missing}
  if ismissing(cardinal_point)
    return missing
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
  data, header = readdlm("weatherAUS.csv", ',', header=true)
  dataframe = DataFrame(data, vec(header))

  if lines > 0
    dataframe = dataframe |> x -> first(x, lines)
  end
  # primeiro, dataframe sem os spikes...
  dataframe = dropmissing!(dataframe)
  select!(dataframe, Not([:Location, :WindDir3pm, :WindDir9am, :WindGustDir]))

  dataframe.Date = map(DateTime, dataframe.Date)
  dataframe.Date = map(datetime2unix, dataframe.Date)
  dataframe.RainToday = dataframe.RainToday .== "Yes"
  dataframe.RainTomorrow = dataframe.RainTomorrow .== "Yes"

  dataframe = select(dataframe, [:MinTemp, :MaxTemp, :RainTomorrow])
  dataframe = filter(row -> !any(x -> x == "NA", row), eachrow(dataframe))

  #for categoricalColumn in [:Location, :WindDir3pm, :WindDir9am, :WindGustDir]
  #  dataframe[!, categoricalColumn] = categorical(dataframe[!, categoricalColumn])
  #end

  return dataframe
end

data = DataFrame(readAndProcessData(50000))
deltaTemp = data.MaxTemp .- data.MinTemp
data.DeltaTemp = deltaTemp
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
for epoch in 1:100#_000
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

print(length(correct), "/", length(data_test), " corretos")

#end