using Test
using Logging
using TerraformingAgents
using Agents
using Random
using DrWatson: @dict
using Suppressor: @suppress_err
using DelimitedFiles
@testset "All test files" begin
    include("TerraformingAgents.jl")
end
