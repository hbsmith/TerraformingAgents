module TerraformingAgents;

using Agents, Random
using Statistics: cor
using DrWatson: @dict, @unpack
using Suppressor: @suppress_err
using LinearAlgebra: dot, diag, issymmetric, tril!
using Distributions: Uniform
using NearestNeighbors
using Distances

export Planet, Life, galaxy_model_setup, galaxy_model_step!, GalaxyParameters, filter_agents

"""
    direction(start::AbstractAgent, finish::AbstractAgent)

Returns normalized direction from `start::AbstractAgent` to `finish::AbstractAgent` (not
user facing).
"""
direction(start::AbstractAgent, finish::AbstractAgent) = let δ = finish.pos .- start.pos
    δ ./ hypot(δ...)
end

Base.@kwdef mutable struct Planet <: AbstractAgent
    id::Int
    pos::NTuple{2, Float64}
    vel::NTuple{2, Float64}

    composition::Vector{Int} ## Represents the planet's genotype
    initialcomposition::Vector{Int} = composition ## Same as composition until it's terraformed

    alive::Bool = false
    ## True if any Life has this planet as its destination
    claimed::Bool = false

    # Properties of the process, but not the planet itself

    ## Planets that phylogenetically preceded this one
    ancestors::Vector{Planet} = Planet[]

    ## Planet that directly preceded this one
    parentplanet::Union{Planet, Nothing} = nothing
    parentlife::Union{<:AbstractAgent, Nothing} = nothing
    parentcomposition::Union{Vector{Int}, Nothing} = nothing
end

Base.@kwdef mutable struct Life <: AbstractAgent
    id::Int
    pos::NTuple{2, Float64}
    vel::NTuple{2, Float64}
    parentplanet::Planet
    composition::Vector{Int} ## Taken from parentplanet
    destination::Union{Planet, Nothing}
    ancestors::Vector{Life} ## Life agents that phylogenetically preceded this one
end

function random_positions(rng, (xmax, ymax), n)
    Ux = Uniform(0, xmax)
    Uy = Uniform(0, ymax)
    collect(zip(rand(rng, Ux, n), rand(rng, Uy, n))) :: Vector{NTuple{2, Float64}}
end

default_velocities(n) = fill((0.0, 0.0), n) :: Vector{NTuple{2, Float64}}

random_compositions(rng, maxcomp, compsize, n) = rand(rng, 1:maxcomp, compsize, n)

random_radius(rng, rmin, rmax) = sqrt(rand(rng) * (rmax^2 - rmin^2) + rmin^2)

filter_agents(model,agenttype) = filter(kv->kv.second isa agenttype, model.agents)

"""
    All get passed to the ABM model as ABM model properties

maxcomp is used for any planets that are not specified when the model is initialized.
compsize must match any compositions provided.
"""
mutable struct GalaxyParameters
    rng # not part of ABMkwargs because it can previously be used for other things
    extent # not part of SpaceArgs because it can previously be used for other things
    ABMkwargs
    SpaceArgs
    SpaceKwargs
    dt
    lifespeed
    interaction_radius
    allowed_diff
    ool
    pos
    vel
    maxcomp
    compsize
    planetcompositions    

    function GalaxyParameters(;
        rng::AbstractRNG = Random.default_rng(),
        extent::NTuple{2,<:Real} = (1.0, 1.0), 
        ABMkwargs::Union{Dict{Symbol},Nothing} = nothing,
        SpaceArgs::Union{Dict{Symbol},Nothing} = nothing,
        SpaceKwargs::Union{Dict{Symbol},Nothing} = nothing,
        dt::Real = 10,
        lifespeed::Real = 0.2,
        interaction_radius::Real = dt*lifespeed,
        allowed_diff::Real = 2.0,
        ool::Union{Vector{Int}, Int, Nothing} = nothing,
        pos::Vector{<:NTuple{2, <:Real}},
        vel::Vector{<:NTuple{2, <:Real}},
        maxcomp::Int,
        compsize::Int,
        planetcompositions::Array{<:Int, 2})

        if !(length(pos) == length(vel) == size(planetcompositions, 2))
            throw(ArgumentError("keyword arguments :pos and :vel must have the same length as the width of :planetcompositions"))
        end

        if ~all(x->length(x)==compsize, eachcol(planetcompositions))
            throw(ArgumentError("All planets compositions must have length of `compsize`"))
        end
        
        ## ABMkwargs
        if ABMkwargs === nothing 
            ABMkwargs = Dict(:rng => rng, :warn => false)
        elseif :rng in ABMkwargs
            rng != ABMkwargs[:rng] && throw(ArgumentError("rng and ABMkwargs[:rng] do not match. ABMkwargs[:rng] will inherit from rng if ABMkwargs[:rng] not provided."))
        else
            ABMkwargs[:rng] = rng
        end

        ## SpaceArgs
        if SpaceArgs === nothing
            SpaceArgs = Dict(:extent => extent)
        elseif :extent in SpaceArgs
            extent != SpaceArgs[:extent] && throw(ArgumentError("extent and SpaceArgs[:extent] do not match. SpaceArgs[:extent] will inherit from extent if SpaceArgs[:extent] not provided."))
        else
            SpaceArgs[:extent] = extent
        end

        ## SpaceKwargs
        SpaceKwargs === nothing && (SpaceKwargs = Dict(:periodic => true))
        
        new(rng, extent, ABMkwargs, SpaceArgs, SpaceKwargs, dt, lifespeed, interaction_radius, allowed_diff, ool, pos, vel, maxcomp, compsize, planetcompositions)

    end
    
end


## Requires one of pos, vel, planetcompositions
## Would it be more clear to write this as 3 separate functions?
function GalaxyParameters(rng::AbstractRNG;
    pos::Union{<:Vector{<:NTuple{2, <:Real}}, Nothing} = nothing,
    vel::Union{<:Vector{<:NTuple{2, <:Real}}, Nothing} = nothing,
    planetcompositions::Union{<:Array{<:Integer,2}, Nothing} = nothing,
    kwargs...)

    # println("rng;")

    if !isnothing(pos)
        nplanets = length(pos)
    elseif !isnothing(vel)
        nplanets = length(vel)
    elseif !isnothing(planetcompositions)
        nplanets = size(planetcompositions, 2)
    else
        msg = "at least one of :pos, :vel or :planetcompositions must be provided as a kwarg"
        throw(ArgumentError(msg))
    end

    args = Dict{Symbol,Any}(kwargs)

    !isnothing(pos) && (args[:pos] = pos)
    !isnothing(vel) && (args[:vel] = vel)
    !isnothing(planetcompositions) && (args[:planetcompositions] = planetcompositions)

    ## Uses GalaxyParameters(rng::AbstractRNG, nplanets::Int; ...) constructor for other arguments
    GalaxyParameters(rng, nplanets; args...)
end

function GalaxyParameters(nplanets::Int; kwargs...)
    ## Uses GalaxyParameters(rng::AbstractRNG, nplanets::Int; ...) constructor for other arguments
    GalaxyParameters(Random.default_rng(), nplanets; kwargs...) ## If it's ", kwargs..." instead of "; kwargs...", then I get an error from running something like this: TerraformingAgents.GalaxyParameters(1;extent=(1.0,1.0))
end

## Main external constructor for GalaxyParameters (other external constructors call it)
## Create with random pos, vel, planetcompositions
## Properties of randomized planetcompositions can be changed with new fields maxcomp, compsize
function GalaxyParameters(rng::AbstractRNG, nplanets::Int;
    extent=(1.0, 1.0), 
    maxcomp=10, 
    compsize=10,
    pos=random_positions(rng, extent, nplanets),
    vel=default_velocities(nplanets),
    planetcompositions=random_compositions(rng, maxcomp, compsize, nplanets),
    kwargs...)

    ## Calls the internal constructor. I still don't understand how this works and passes the correct keyword arguments to the correct places
    GalaxyParameters(; rng, extent, pos, vel, maxcomp, compsize, planetcompositions, kwargs...)
end



# """
#     Other kwargs of ABM besides properties (which fall into GalaxyProperties above)
#     Do I need this or can I wrap it into some other kwargs so that I don't have to copy and paste default kwargs?
# """
# struct ABMkwargs
#     scheduler
#     rng
#     warn

#     function ABMkwargs(
#         scheduler = Schedulers.fastest
#         rng::AbstractRNG = Random.default_rng()
#         warn = false)

#         new(scheduler, rng, warn)

#     end
# end

# """
#     All get passed into ContinuousSpace
# """
# struct SpaceArgs
#     extent
#     spacing
#     update_vel!
#     periodic

#     function SpaceArgs(
#         extent::NTuple{2,<:Real}=(1.0, 1.0),
#         spacing = min(extent...) / 10.0;
#         update_vel! = defvel,
#         periodic = true)

#         new(extent, spacing, update_vel!, periodic)
#     end

# end

nplanets(params::GalaxyParameters) = length(params.pos)

center_position(pos::NTuple{2, <:Real}, extent::NTuple{2, <:Real}, m::Real) = pos.+((extent.-(extent./m))./2) #pos.+(

"""
    galaxy_model_setup(params::GalaxyParameters)

Sets up the galaxy model.
"""
function galaxy_model_setup(params::GalaxyParameters)

    model = galaxy_planet_setup(params)
    model = galaxy_life_setup(model, params::GalaxyParameters)
    model

end

function galaxy_planet_setup(params::GalaxyParameters)

    extent_multiplier = 3
    params.extent = extent_multiplier.*params.extent

    if :spacing in keys(params.SpaceArgs)
        space2d = ContinuousSpace(params.extent, params.SpaceArgs[:spacing]; params.SpaceKwargs...)
    else
        space2d = ContinuousSpace(params.extent; params.SpaceKwargs...)
    end

    model = @suppress_err AgentBasedModel(
        Union{Planet,Life},
        space2d,
        properties = Dict(:dt => params.dt,
                        :lifespeed => params.lifespeed,
                        :interaction_radius => params.interaction_radius,
                        :allowed_diff => params.allowed_diff,
                        :nplanets => nplanets(params),
                        :maxcomp => params.maxcomp,
                        :compsize => params.compsize,
                        :GalaxyParameters => params);
                        # :nlife => length(params.ool)
                        # :ool => params.ool,
                        # :pos => params.pos,
                        # :vel => params.vel,
                        # :planetcompositions => params.planetcompositions); ## Why does having a semicolon here fix it???
        # rng=params.ABMkwargs[:rng],
        # warn=params.ABMkwargs[:warn]
        params.ABMkwargs... ## Why does this not work??
    )

    initialize_planets!(model, params, extent_multiplier)
    model

end

function galaxy_life_setup(model, params::GalaxyParameters)

    agent = isnothing(params.ool) ? random_agent(model, x -> x isa Planet) : model.agents[params.ool]
    spawnlife!(agent, model)
    # index!(model)
    model

end


# galaxy_model_setup(params::GalaxyParameters) = galaxy_model_setup(params)

# function galaxy_model_setup(rng::AbstractRNG, args...; kwargs...)
#     galaxy_model_setup(rng, GalaxyParameters(rng, args..., kwargs...))
# end

"""
    initialize_planets!(model, params::GalaxyParameters)

Adds Planets (not user facing).

Called by [`galaxy_model_setup`](@ref).
"""
function initialize_planets!(model, params::GalaxyParameters, extent_multiplier)
    for i = 1:nplanets(params)
        id = nextid(model)
        pos = center_position(params.pos[i], params.extent, extent_multiplier)
        vel = params.vel[i]
        composition = params.planetcompositions[:, i]

        planet = Planet(; id, pos, vel, composition)

        add_agent_pos!(planet, model)
    end
    model
end

"""
    compatibleplanets(planet::Planet, model::ABM)

Returns `Vector{Planet}` of `Planet`s compatible with `planet` for terraformation (not user
facing).
"""
function compatibleplanets(planet::Planet, model::ABM)
    function iscandidate((_, p))
        isa(p, Planet) && !p.alive && !p.claimed && p.id != planet.id
    end

    candidateplanets = collect(values(filter(iscandidate, model.agents)))
    compositions = hcat([a.composition for a in candidateplanets]...)
    compositiondiffs = abs.(compositions .- planet.composition)
    compatibleindxs =
        findall(<=(model.allowed_diff), vec(maximum(compositiondiffs, dims = 1)))

    ## Necessary in cased the result is empty
    convert(Vector{Planet}, candidateplanets[compatibleindxs]) ## Returns Planets

end

"""
    nearestcompatibleplanet(planet::Planet, candidateplanets::Vector{PLanet})

Returns `Planet` within `candidateplanets` that is nearest to `planet `(not user facing).
"""
function nearestcompatibleplanet(planet::Planet, candidateplanets::Vector{Planet})

    length(candidateplanets) == 0 && throw(ArgumentError("candidateplanets is empty"))
    planetpositions = Array{Float64}(undef, 2, length(candidateplanets))
    for (i, a) in enumerate(candidateplanets)
        planetpositions[1, i] = a.pos[1]
        planetpositions[2, i] = a.pos[2]
    end
    idx, dist = nn(KDTree(planetpositions), collect(planet.pos))
    candidateplanets[idx] ## Returns Planet

end

"""
    spawnlife!(planet::Planet, model::ABM; ancestors::Vector{Life} = Life[])

Spawns `Life` (not user facing).

Called by [`galaxy_model_setup`](@ref).
"""
function spawnlife!(
    planet::Planet,
    model::ABM;
    ancestors::Vector{Life} = Life[]
    )

    planet.alive = true
    planet.claimed = true ## This should already be true unless this is the first planet
    ## No ancestors, parentplanet, parentlife, parentcomposition
    candidateplanets = compatibleplanets(planet, model)
    if length(candidateplanets) == 0
        @warn "Life on Planet $(planet.id) has no compatible planets. It's the end of its line."
        destinationplanet = nothing
        vel = planet.pos .* 0.0
    else
        destinationplanet = nearestcompatibleplanet(planet, candidateplanets)
        vel = direction(planet, destinationplanet) .* model.lifespeed
    end

    life = Life(;
        id = nextid(model),
        pos = planet.pos,
        vel = vel,
        parentplanet = planet,
        composition = planet.composition,
        destination = destinationplanet,
        ancestors
    ) ## Only "first" life won't have ancestors

    life = add_agent_pos!(life, model)

    !isnothing(destinationplanet) && (destinationplanet.claimed = true) ## destination is only nothing if no compatible planets
    ## NEED TO MAKE SURE THAT THE FIRST LIFE HAS PROPERTIES RECORDED ON THE FIRST PLANET

    model
end

"""
    mixcompositions(lifecomposition::Vector{Int}, planetcomposition::Vector{Int})

Rounds element-averaged composition (not user facing).
"""
function mixcompositions(lifecomposition::Vector{Int}, planetcomposition::Vector{Int})
    ## Simple for now; Rounding goes to nearest even number
    round.(Int, (lifecomposition .+ planetcomposition) ./ 2)
end

"""
    terraform!(life::Life, planet::Planet, model::ABM)

Performs actions on `life` and `planet` associated with successful terraformation. Takes
existing `life` and terraforms an exsiting non-alive `planet` (not user facing).
- Mix the `composition` of `planet` and `life`
- Update the `planet` to `alive=true`
- Update the `planet`'s `ancestors`, `parentplanet`, `parentlife`, and `parentcomposition`
- Call `spawnlife!` to send out `Life` from `planet`.
"""
function terraform!(life::Life, planet::Planet, model::ABM)

    ## Modify destination planet properties
    planet.composition = mixcompositions(planet.composition, life.composition)
    planet.alive = true
    push!(planet.ancestors, life.parentplanet)
    planet.parentplanet = life.parentplanet
    planet.parentlife = life
    planet.parentcomposition = life.composition
    # planet.claimed = true ## Test to make sure this is already true beforehand

    spawnlife!(planet, model, ancestors = push!(life.ancestors, life)) ## This makes new life
end

"""
    update_planets_and_life(model::ABM)

Checks pairs of planets, identifying which should be terraformed. Kills life that
terraforms.  
"""
function update_planets_and_life!(model::ABM)
    ## I need to scale the interaction radius by dt and the velocity of life or else I can
    ##   miss some interactions
    
    life_to_kill = Life[]
    for (a1, a2) in interacting_pairs(model, model.interaction_radius, :types, nearby_f = nearby_ids_exact)
        life, planet = typeof(a1) == Planet ? (a2, a1) : (a1, a2)
        if planet == life.destination
            terraform!(life, planet, model)
            push!(life_to_kill, life)
        end

    end

    for life in life_to_kill
        kill_agent!(life, model)
    end

    model

end

"""
    pos_is_inside_alive_radius(pos::Tuple, model::ABM)

Returns false if provided position lies within any life's interaction radii    
"""
function pos_is_inside_alive_radius(pos::Tuple, model::ABM, exact=true)

    if exact==true
        neighbor_ids = collect(nearby_ids_exact(pos,model,model.interaction_radius))
    else
        neighbor_ids = collect(nearby_ids(pos,model,model.interaction_radius))
    end

    if length(filter(kv -> kv.first in neighbor_ids && kv.second isa Planet && kv.second.alive, model.agents)) > 0
        return true
    else
        return false
    end

end

"""
    add_planet!(model::ABM, min_dist, max_dist, max_attempts)

Adds a planet to the galaxy that is within the interaction radius of a non-living planet,
and outside the interaction radius of all living planets. Max attempts sets the limit of
iterations in the while loop to find a valid planet position (default = 10*nplanets).

"""
function add_planet!(model::ABM, 
    min_dist=model.interaction_radius/10, 
    max_dist=model.interaction_radius, 
    max_attempts=10*length(filter(kv -> kv.second isa Planet, model.agents))
)

    id = nextid(model)

    ## https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
    n_attempts = 0
    valid_pos = false
    while valid_pos == false && n_attempts < max_attempts
        r = random_radius(model.rng, min_dist, max_dist)
        theta = rand(model.rng)*2*π

        for (_, planet) in Random.shuffle(model.rng, collect(filter(kv -> kv.second isa Planet && ~kv.second.alive, model.agents)))
            pos = (planet.pos[1] + r*cos(theta), planet.pos[2] + r*sin(theta))
            if length(collect(nearby_ids_exact(pos,model,min_dist))) == 0 && ~pos_is_inside_alive_radius(pos,model)
                valid_pos = true
                vel = default_velocities(1)[1] 
                composition = vec(random_compositions(model.rng, model.maxcomp, model.compsize, 1))
                newplanet = Planet(; id, pos, vel, composition)
                add_agent_pos!(newplanet, model)
                println("Planet added at $pos")
                return model
            end

            n_attempts += 1

        end

    end

    println("Planet unable to be added in valid position within `max_attempts`")

    model

end

"""
    update_nplanets!(model::ABM)

Adds planets to the model at random positions.
"""
function update_nplanets!(model)
    while model.properties[:nplanets] > length(filter(kv->kv.second isa Planet, model.agents))
        add_planet!(model::ABM)
    end
end

"""
    galaxy_model_step(model)

Custom `model_step` to be called by `Agents.step!`. Checks all `interacting_pairs`, and
`terraform`s a `Planet` if a `Life` has reached its destination; then kills that `Life`.
"""
function galaxy_model_step!(model)
    
    update_planets_and_life!(model)
    update_nplanets!(model)

end

#######################################
## Distances, Correlations and permutations
"""

    concatenate_planet_fields(field, model, planet_condition=nothing)

Concatenate all the of a model's planet's fields into a single m x n matrix.

Planets are filtered by the provided optional `planet_condition` arg.

e.g. Concatonate the :composition of every planet into a single matrix.

"""
function concatenate_planet_fields(field, model, planet_condition=nothing)

    field_values = []
    for (id,planet) in filter_agents(model,Planet)

        if planet_condition != nothing             
            planet_condition(planet) && push!(field_values, collect(getfield(planet, field)))
        else
            push!(field_values, collect(getfield(planet, field)))
        end

    end

    hcat(field_values...)

end

# function planet_distance_matrix(field, model, dist_metric=Euclidean(), planet_condition=nothing)
    
#     pairwise(dist_metric, hcat(field_values...), dims=2)

# end

# function position_distance_matrix(dist_metric, model, planet_condition)

#     positions = [collect(kv.second.pos) for kv in filter_agents(model,Planet)]
#     pairwise(dist_metric, hcat(positions...), dims=2)
# end
"""

    PlanetMantelTest(model, xfield=:composition, yfield=:pos, dist_metric=Euclidean();  method=:pearson, permutations=999, alternative=:twosided, planet_condition=nothing)

xfield::symbol is meant to be one of (:composition, :pos)
yfield::symbol is meant to be one of (:composition, :pos)
dist_metric::dist from Distances.jl (default: Euclidean())
planet_condition takes an optional function which can be used to filter planets. For example, filtering by only planets which are alive. 
"""
function PlanetMantelTest(model, xfield=:composition, yfield=:pos; dist_metric=Euclidean(),  method=:pearson, permutations=999, alternative=:twosided, planet_condition=nothing)

    x = pairwise(dist_metric, concatenate_planet_fields(xfield, model, planet_condition), dims=2)
    y = pairwise(dist_metric, concatenate_planet_fields(yfield, model, planet_condition), dims=2)

    MantelTest(x, y;  rng=model.rng, dist_metric=dist_metric, method=method, permutations=permutations, alternative=alternative)

end

function MantelTest(x, y;  rng::AbstractRNG = Random.default_rng(), dist_metric=Euclidean(), method=:pearson, permutations=999, alternative=:twosided)
    ## Based this off of https://github.com/biocore/scikit-bio/blob/0.1.3/skbio/math/stats/distance/_mantel.py#L23

    method == :pearson ? corr_func = cor : throw(ArgumentError("Not yet implemented")) 

    permutations < 0 && throw(ArgumentError("Number of permutations must be greater than or equal to zero."))
    alternative ∉ [:twosided, :greater, :less] && throw(ArgumentError("Invalid alternative hypothesis $alternative."))

    ## Below not needed because x and y must already be distance matrices here
    # x = pairwise(dist_metric, x, dims=2)
    # y = pairwise(dist_metric, y, dims=2)

    ## Check properties to verify they're distance matrices
    size(x) != size(y) && throw(ArgumentError("Distance matrices must have the same shape."))
    size(x)[1] < 3 && throw(ArgumentError("Distance matrices must be at least 3x3 in size."))
    sum(abs.(diag(x))) + sum(abs.(diag(y))) != 0 && throw(ArgumentError("Distance matrices must be hollow. I think."))
    ~issymmetric(x) | ~issymmetric(y) && throw(ArgumentError("Distance matrices must be symmetric. I think."))

    ## This part just needs to get a flattened version of the diagonal of a hollow, square, symmetric matrix
    x_flat = x[tril!(trues(size(x)), -1)]
    y_flat = y[tril!(trues(size(y)), -1)]

    orig_stat = corr_func(x_flat, y_flat)

    ## Permutation tests
    if (permutations == 0) | isnan(orig_stat)
        p_value = NaN
    else
        perm_gen = (corr_func(Random.shuffle(rng, x_flat), y_flat) for _ in 1:permutations)
        permuted_stats = collect(Iterators.flatten(perm_gen))

        if alternative == :twosided
            count_better = sum(abs.(permuted_stats) .>= abs(orig_stat))
        elseif alternative == :greater
            count_better = sum(permuted_stats .>= orig_stat)
        else
            count_better = sum(permuted_stats .<= orig_stat)
        end

        p_value = (count_better + 1) / (permutations + 1)
        
    end

    return orig_stat, p_value

end



# rng = MersenneTwister(3141)
# x = [[0,1,2],[1,0,3],[2,3,0]]
# y = [[0, 2, 7],[2, 0, 6],[7, 6, 0]]
# MantelTest(hcat(x...),hcat(y...), rng=rng)

## Fun with colors
# col_to_hex(col) = "#"*hex(col)
# hex_to_col(hex) = convert(RGB{Float64}, parse(Colorant, hex))
# mix_cols(c1, c2) = RGB{Float64}((c1.r+c2.r)/2, (c1.g+c2.g)/2, (c1.b+c2.b)/2)


end # module
