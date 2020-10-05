module TerraformingAgents

using Agents, Random, AgentsPlots, Plots
using DrWatson: @dict, @unpack
using Suppressor: @suppress_err
using LinearAlgebra: dot
using NearestNeighbors

export 
Planet, 
Life,
galaxy_model_basic,
galaxy_model_advanced,
galaxy_model_step!

Agents.random_agent(model, A::Type{T}, RNG::AbstractRNG=Random.default_rng()) where {T<:AbstractAgent} = model[rand(RNG, [k for (k,v) in model.agents if v isa A])]

magnitude(x::Tuple{<:Real,<:Real}) = sqrt(sum(x .^ 2))
direction(start::AbstractAgent, finish::AbstractAgent) = (finish.pos .- start.pos) ./ magnitude((finish.pos .- start.pos)) ## This is normalized

Base.@kwdef mutable struct Planet <: AbstractAgent
    id::Int
    pos::NTuple{2,<:AbstractFloat}
    vel::NTuple{2,<:AbstractFloat}
    
    composition::Vector{Int} ## Represents the planet's genotype
    initialcomposition::Vector{Int} ## Same as composition until it's terraformed
    alive::Bool
    claimed::Bool ## If life has the planet as its destination
    
    ## Properties of the process, but not the planet itself
    ancestors::Vector{Planet} ## List of all planets that phylogenetically preceded this life
    parentplanet::Union{Planet,Nothing} 
    parentlife::Union{<:AbstractAgent,Nothing} ## This shouild be of type Life, but I can't force because it would cause a mutually recursive declaration b/w Planet and Life
    parentcomposition::Union{Vector{Int},Nothing} 

end

Base.@kwdef mutable struct Life <: AbstractAgent
    id::Int
    pos::NTuple{2,<:AbstractFloat}
    vel::NTuple{2,<:AbstractFloat}
    parentplanet::Planet 
    composition::Vector{Int} ## Represents the parent planet's genotype
    destination::Union{Nothing,Planet} 
    ancestors::Vector{Life} ## List of all life that phylogenetically preceded this life
end

function galaxy_model_setup(detail::Symbol, kwarg_dict::Dict)

    @unpack RNG, extent, dt, interaction_radius, allowed_diff, lifespeed = kwarg_dict

    space2d = ContinuousSpace(2; periodic = true, extend = extent)
    model = @suppress_err AgentBasedModel(
        Union{Planet,Life}, 
        space2d, 
        properties = @dict(
            dt, 
            interaction_radius, 
            allowed_diff, 
            lifespeed))

    if detail == :basic 
        @unpack nplanets = kwarg_dict
        initialize_planetarysystems_basic!(model, nplanets; @dict(RNG)...)
    elseif detail == :advanced
        @unpack pos, vel, planetcompositions = kwarg_dict
        initialize_planetarysystems_advanced!(model; @dict(RNG, pos, vel, planetcompositions)...)
    else
        throw(ArgumentError("`detail` must be `:basic` or `:advanced`"))
    end
    
    initialize_life!(random_agent(model, Planet, RNG), model)   
    index!(model)
    model

end

function galaxy_model_basic(
    nplanets::Int; 
    RNG::AbstractRNG=Random.default_rng(), 
    extent::Tuple{<:Real,<:Real} = (1,1), ## Size of space
    dt::Real = 1.0, 
    interaction_radius::Real = 0.02, ## How close life and destination planet have to be to interact
    allowed_diff::Real = 3, ## How similar each element of life and destination planet have to be for terraformation
    lifespeed::Real = 0.2) ## Speed that life spreads

    galaxy_model_setup(:basic, @dict(nplanets, RNG, extent, dt, interaction_radius, allowed_diff, lifespeed))

end

function galaxy_model_advanced(; 
    RNG::AbstractRNG=Random.default_rng(), 
    extent::Tuple{<:Real,<:Real} = (1,1), 
    dt::Real = 1.0, 
    interaction_radius::Real = 0.02, 
    allowed_diff::Real = 3, 
    lifespeed::Real = 0.2,
    pos::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    vel::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    planetcompositions::Union{Nothing,Vector{Vector{Int}}} = nothing)

    galaxy_model_setup(:advanced, @dict(RNG, extent, dt, interaction_radius, allowed_diff, pos, vel, planetcompositions, lifespeed))

end

function providedargs(args::Dict) 
    
    providedargs = filter(x -> !isnothing(x.second), args)

    isempty(providedargs) ? throw(ArgumentError("one of $(keys(args)) must be provided")) : providedargs

end 

haveidenticallengths(args::Dict) = all(length(i.second) == length(args[collect(keys(args))[1]]) for i in args)

function initialize_planetarysystems_unsafe!(
    model::AgentBasedModel,
    nplanets::Int; 
    RNG::AbstractRNG = Random.default_rng(),
    pos::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    vel::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    planetcompositions::Union{Nothing,Vector{Vector{Int}}} = nothing)

    ## Initialize arguments which are not provided 
    ## (flat random pos, no velocity, flat random compositions, 1 planet per system)
    isnothing(pos) && (pos = [Tuple(rand(RNG,2)) for _ in 1:nplanets])
    isnothing(vel) && (vel = [(0,0) for _ in 1:nplanets])
    isnothing(planetcompositions) && (planetcompositions = [rand(RNG,1:10,10) for _ in 1:nplanets])

    for i in 1:nplanets
        
        pskwargs = Dict(:id => nextid(model),
                    :pos => pos[i],
                    :vel => vel[i],
                    :composition => planetcompositions[i],
                    :initialcomposition => planetcompositions[i],
                    :alive => false,
                    :claimed => false,
                    :parentplanet => nothing,
                    :parentlife => nothing,
                    :parentcomposition => nothing,
                    :ancestors => Vector{Planet}(undef,0))
        
        add_agent_pos!(Planet(;pskwargs...), model)
    
    end
    model

end

function initialize_planetarysystems_basic!(
    model::AgentBasedModel,
    nplanets::Int; 
    RNG::AbstractRNG = Random.default_rng())

    nplanets < 1 && throw(ArgumentError("At least one planetary system required."))

    pos=nothing
    vel=nothing 
    planetcompositions=nothing

    initialize_planetarysystems_unsafe!(model, nplanets; @dict(RNG, pos, vel, planetcompositions)...)    

end

function initialize_planetarysystems_advanced!(
    model::AgentBasedModel; 
    RNG::AbstractRNG = Random.default_rng(),
    pos::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    vel::Union{Nothing,AbstractArray{<:NTuple{2,<:AbstractFloat}}} = nothing,
    planetcompositions::Union{Nothing,Vector{Vector{Int}}} = nothing)

    ## Validate user's args
    userargs = providedargs(@dict(pos, vel, planetcompositions))
    haveidenticallengths(userargs) || throw(ArgumentError("provided arguments $(keys(userargs)) must all be same length"))
    
    ## Infered from userargs
    nplanets = length(userargs[collect(keys(userargs))[1]])

    initialize_planetarysystems_unsafe!(model, nplanets; @dict(RNG, pos, vel, planetcompositions)...)    

end

function compatibleplanets(planet::Planet, allowed_diff::Real)

    cadidateplanets = collect(values(filter(p->p.second.alive==false & isa(p.second, Planet) & p.second.claimed==false, model.agents)))
    compositions = hcat([a.composition for a in candidateplanets]...)
    compositiondiffs = abs.(compositions .- planet.composition)
    compatibleindxs = findall(<=(allowed_diff),maximum(compositiondiffs, dims=1))
    candidateplanets[compatibleindxs] ## Returns Planet

end

function nearestcompatibleplanet(planet::Planet, compatibleplanets::Vector{Planet})

    planetpositions = Array{Real}(undef,2,length(compatibleplanets))
    for (i,a) in enumerate(compatibleplanets)
        planetpositions[1,i] = a.pos[1]
        planetpositions[2,i] = a.pos[2]
    end
    idx, dist = nn(KDTree(planetpositions),collect(planet.pos)) 
    compatibleplanets[idx] ## Returns Planet

end

function spawnlife!(planet::Planet, model::ABM; ancestors::Union{Nothing,Vector{Life}}=nothing) 
    ## Design choice is to modify planet and life together since the life is just a reflection of the planet anyways
    planet.alive = true
    planet.claimed = true ## This should already be true unless this is the first planet
    ## No ancestors, parentplanet, parentlife, parentcomposition
    destinationplanet =  nearestcompatibleplanet(planet, compatibleplanets(planet, model.allowed_diff))

    args = Dict(:id => nextid(model),
                :pos => planet.pos,
                :vel => direction(planet, destinationplanet) .* model.lifespeed,
                :parentplanet => planet,
                :parentcomposition => planet.composition,
                :destination => destinationplanet,
                :ancestors => isnothing(ancestor) ? Planet[] : ancestors) ## Only "first" life won't have ancestors

    life = add_agent_pos!(Life(;args...), model)
    
    destinationplanet.claimed = true
    model

end

function mixcompositions(lifecomposition::Vector{Int}, planetcomposition::Vector{Int})
    ## Simple for now
    convert(Vector{Int}, round.(lifecomposition .+ planetcomposition))
end


## Life which has spawned elsewhere merging with an uninhabited (ie dead) planet
function terraform!(life::Life, planet::Planet)

    ## Modify destination planet properties
    planet.composition = mixcompositions(planet.compositon, life.composition)
    planet.alive = true
    push!(planet.ancestors, life.parentplanet)
    planet.parentplanet = life.parentplanet
    planet.parentlife = life
    planet.parentcomposition = life.composition
    # planet.claimed = true ## Test to make sure this is already true beforehand

    spawnlife!(planet, model, ancestors = push!(life.ancestors, life)) ## This makes new life 
    println("terraformed $(planet.id) from $(life.id)")
    kill_agent!(life, model)

end

# function approaching_planet(life::Life, planet::Planet)
# ## Don't think I need this if I just check that the planet is the destination
#     lifesRelativePos = life.pos .- planet.pos
#     lifesRelativeVel = life.vel

#     dot(lifesRelativePos,lifesRelativeVel) >= 0 ? false : true

# end

function galaxy_model_step!(model)
    ## Interaction radius has to account for the velocity of life and the size of dt to ensure interaction
    for (a1, a2) in interacting_pairs(model, model.interaction_radius, :types)
        life, planet = typeof(a1) == Planet ? (a2, a1) : (a1, a2)
        planet == life.destinationplanet && terraform!(life, planet)
        # life.parentplanet == planet && return ## don't accidentally interact with the parent planet
        # approaching_planet(life, planet) && is_compatible(life, planet, model.allowed_diff) ? terraform!(life, planet) : return
    end
end

## TO DO 10/5/2020
## - UPDATE TESTS

#= I should probably have a function that just scales my planet compositions 
accross the color spectrum instead of baking colors in as the compositions
themselves
=#

## Fun with colors
col_to_hex(col) = "#"*hex(col)
hex_to_col(hex) = convert(RGB{Float64}, parse(Colorant, hex))
mix_cols(c1, c2) = RGB{Float64}((c1.r+c2.r)/2, (c1.g+c2.g)/2, (c1.b+c2.b)/2)



# function sir_agent_step!(agent, model)
#     move_agent!(agent, model, model.dt)
#     update!(agent) # store information in life agent of it terraforming?
#     recover_or_die!(agent, model)
# end

# agent_step!(agent, model) = move_agent!(agent, model, model.dt)

# modelparams = Dict(:RNG => MersenneTwister(1236),
#                    : => .45,
#                    :dt => 0.1)

# model = galaxy_model(;modelparams...)

# model_colors(a) = typeof(a) == PlanetarySystem ? "#2b2b33" : "#338c54"

# e = model.space.extend
# anim = @animate for i in 1:2:100
#     p1 = plotabm(
#         model,
#         as = 5,
#         ac = model_colors,
#         showaxis = false,
#         grid = false,
#         xlims = (0, e[1]),
#         ylims = (0, e[2]),
#     )

#     title!(p1, "step $(i)")
#     step!(model, agent_step!, galaxy_model_step!, 2)
# end


# animation_path = "../output/animation/"
# if !ispath(animation_path)
#     mkpath(animation_path)
# end

# gif(anim, joinpath(animation_path,"terraform_test_death1.gif"), fps = 25)

end # module