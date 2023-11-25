module RainManipulation
using CSV
using DataFrames
using Dates
using Flux
using MLJ
export readAndPreprocessData

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

function readAndPreprocessData(lines::Int64=0)
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

  dataframe = CSV.File("weatherAUS.csv", types=column_types, missingstrings=["NA"]) |> DataFrame
  if lines > 0
    dataframe = dataframe |> x -> first(x, lines)
  end

  dataframe.RainToday .= dataframe.RainToday .== "Yes"
  dataframe.RainTomorrow .= dataframe.RainTomorrow .== "Yes"

  dataframe[!, :Date] = Dates.datetime2unix.(DateTime.(dataframe[!, :Date]))

  #categorical_directions = categorical(["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"], ordered=True)

  for cardinalFeature in [:WindDir9am, :WindDir3pm, :WindGustDir]
    dataframe[!, cardinalFeature] = cardinalToRadians.(dataframe[!, cardinalFeature])
  end

  #dataframe = coalesce.(dataframe, 0)

  return dataframe
end
end