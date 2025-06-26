"""
Utility functions for testing and validation
"""

"""
    validate_nbody_data(nbody_data::NBodyData)

Validate N-body data for consistency and completeness.

# Returns
- `Bool`: True if data is valid
- `Vector{String}`: List of validation warnings/errors
"""
function validate_nbody_data(nbody_data::NBodyData)
    warnings = String[]
    is_valid = true
    
    # Check for missing data
    expected_entries = length(nbody_data.star_ids) * length(nbody_data.timesteps)
    actual_entries = length(nbody_data.positions)
    
    if actual_entries != expected_entries
        push!(warnings, "Missing position data: expected $expected_entries, got $actual_entries")
        is_valid = false
    end
    
    # Check for reasonable position ranges
    all_positions = collect(values(nbody_data.positions))
    if !isempty(all_positions)
        position_magnitudes = [norm(pos) for pos in all_positions]
        max_magnitude = maximum(position_magnitudes)
        
        if max_magnitude > 100.0  # More than 100 kpc seems unreasonable
            push!(warnings, "Large position magnitudes detected (max: $(round(max_magnitude, digits=2)) kpc)")
        end
    end
    
    # Check timestep consistency
    timestep_diffs = diff(sort(nbody_data.timesteps))
    if !all(d -> d == timestep_diffs[1], timestep_diffs)
        push!(warnings, "Inconsistent timestep intervals detected")
    end
    
    # Check for duplicate entries
    if length(unique(collect(keys(nbody_data.positions)))) != length(nbody_data.positions)
        push!(warnings, "Duplicate position entries detected")
        is_valid = false
    end
    
    return is_valid, warnings
end

"""
    analyze_reachability(nbody_data::NBodyData, max_velocity::Float64, 
                        sample_timesteps::Int=10)

Analyze reachability patterns in the N-body data.

# Arguments
- `nbody_data`: N-body trajectory data
- `max_velocity`: Maximum travel velocity in kpc/year
- `sample_timesteps`: Number of timesteps to sample for analysis

# Returns
- `Dict`: Analysis results including connectivity statistics
"""
function analyze_reachability(nbody_data::NBodyData, max_velocity::Float64, 
                            sample_timesteps::Int=10)
    # Sample timesteps for analysis
    n_timesteps = length(nbody_data.timesteps)
    sample_indices = range(1, n_timesteps-1, length=min(sample_timesteps, n_timesteps-1))
    sample_timesteps_list = nbody_data.timesteps[Int.(round.(sample_indices))]
    
    connectivity_stats = []
    
    for timestep in sample_timesteps_list
        # Calculate connectivity matrix for this timestep
        n_stars = length(nbody_data.star_ids)
        reachable_counts = Int[]
        
        for star_id in nbody_data.star_ids
            reachable = calculate_reachable_destinations(
                nbody_data, star_id, timestep, max_velocity, Set{Int}()
            )
            push!(reachable_counts, length(reachable))
        end
        
        # Calculate statistics
        stats = Dict(
            :timestep => timestep,
            :mean_reachable => mean(reachable_counts),
            :max_reachable => maximum(reachable_counts),
            :min_reachable => minimum(reachable_counts),
            :connectivity_fraction => mean(reachable_counts) / (n_stars - 1)
        )
        
        push!(connectivity_stats, stats)
    end
    
    return Dict(
        :connectivity_over_time => connectivity_stats,
        :total_stars => length(nbody_data.star_ids),
        :total_timesteps => length(nbody_data.timesteps),
        :max_velocity => max_velocity
    )
end

"""
    create_test_nbody_data(n_stars::Int=10, n_timesteps::Int=100;
                          cluster_radius::Float64=0.01,
                          timestep_duration::Float64=1000.0)

Create synthetic N-body data for testing purposes.

# Arguments
- `n_stars`: Number of stars
- `n_timesteps`: Number of timesteps
- `cluster_radius`: Radius of star cluster in kpc
- `timestep_duration`: Duration of each timestep in years

# Returns
- `NBodyData`: Synthetic trajectory data
"""
function create_test_nbody_data(n_stars::Int=10, n_timesteps::Int=100;
                               cluster_radius::Float64=0.01,
                               timestep_duration::Float64=1000.0)
    positions = Dict{Tuple{Int,Int}, SVector{3,Float64}}()
    
    # Generate initial random positions within cluster
    initial_positions = [cluster_radius * normalize(randn(3)) * rand()^(1/3) for _ in 1:n_stars]
    
    # Generate simple orbital motion (circular orbits around center)
    orbital_periods = [rand(50:200) for _ in 1:n_stars]  # timesteps per orbit
    
    for timestep in 0:(n_timesteps-1)
        for (star_idx, star_id) in enumerate(1:n_stars)
            # Simple circular motion
            initial_pos = initial_positions[star_idx]
            radius = norm(initial_pos)
            period = orbital_periods[star_idx]
            
            angle = 2π * timestep / period
            rotation_matrix = [cos(angle) -sin(angle) 0;
                             sin(angle)  cos(angle) 0;
                             0           0          1]
            
            new_pos = SVector{3,Float64}(rotation_matrix * initial_pos)
            positions[(star_id, timestep)] = new_pos
        end
    end
    
    timesteps = collect(0:(n_timesteps-1))
    star_ids = collect(1:n_stars)
    
    return NBodyData(positions, timesteps, star_ids, timestep_duration)
end

"""
    run_quick_test(csv_path::Union{String,Nothing}=nothing)

Run a quick test of the N-body integration system.

# Arguments
- `csv_path`: Path to N-body CSV file, or nothing to use synthetic data

# Returns
- `Dict`: Test results
"""
function run_quick_test(csv_path::Union{String,Nothing}=nothing)
    println("=== N-Body Integration Quick Test ===")
    
    # Load or create test data
    if csv_path !== nothing
        println("Loading N-body data from: $csv_path")
        nbody_data = load_nbody_data(csv_path)
    else
        println("Creating synthetic test data...")
        nbody_data = create_test_nbody_data(20, 50)  # 20 stars, 50 timesteps
    end
    
    # Validate data
    println("\nValidating data...")
    is_valid, warnings = validate_nbody_data(nbody_data)
    if !is_valid
        println("❌ Data validation failed:")
        for warning in warnings
            println("  - $warning")
        end
        return Dict(:success => false, :errors => warnings)
    else
        println("✅ Data validation passed")
        if !isempty(warnings)
            println("Warnings:")
            for warning in warnings
                println("  - $warning")
            end
        end
    end
    
    # Analyze reachability
    println("\nAnalyzing reachability...")
    test_velocity = 0.001  # kpc/year
    reachability = analyze_reachability(nbody_data, test_velocity, 5)
    println("Mean connectivity: $(round(mean([s[:connectivity_fraction] for s in reachability[:connectivity_over_time]]), digits=3))")
    
    # Setup and run short simulation
    println("\nRunning short simulation...")
    try
        model = setup_nbody_model(nbody_data; max_velocity=test_velocity, nool=2)
        final_model, step_data = run_nbody_simulation(nbody_data, min(10, length(nbody_data.timesteps)-1))
        
        println("✅ Simulation completed successfully")
        println("Final living planets: $(final_model.n_living_planets)")
        println("Total terraforming events: $(sum([s[:terraformed] for s in step_data]))")
        
        return Dict(
            :success => true,
            :n_stars => length(nbody_data.star_ids),
            :n_timesteps => length(nbody_data.timesteps),
            :final_living_planets => final_model.n_living_planets,
            :reachability => reachability,
            :step_data => step_data
        )
        
    catch e
        println("❌ Simulation failed: $e")
        return Dict(:success => false, :error => string(e))
    end
end

# Export utility functions  
export validate_nbody_data, analyze_reachability, create_test_nbody_data, run_quick_test

# Export all main functions
export NBodyData, load_nbody_data, get_position, has_position,
       calculate_reachable_destinations, select_most_similar_destination,
       plan_mission_nbody, LifeMission, create_mission, process_arriving_missions!, 
       terraform_from_mission!, attempt_launch_missions!, can_launch_mission,
       NBodyPlanet, setup_nbody_model, nbody_agent_step!, nbody_model_step!, 
       run_nbody_simulation

"""
N-Body Data Integration for TerraformingAgents.jl

This module provides functionality to integrate N-body simulation data
with the agent-based terraforming simulation.
"""

using CSV
using DataFrames
using StaticArrays
using Distances

# Data structure to hold N-body trajectory data
struct NBodyData
    # Core position data indexed by (star_id, timestep)
    positions::Dict{Tuple{Int,Int}, SVector{3,Float64}}
    
    # Metadata
    timesteps::Vector{Int}
    star_ids::Vector{Int}
    timestep_duration::Float64  # Duration of each timestep in years
    max_timestep::Int
    
    function NBodyData(positions, timesteps, star_ids, timestep_duration)
        max_timestep = maximum(timesteps)
        new(positions, sort(timesteps), sort(star_ids), timestep_duration, max_timestep)
    end
end

"""
    load_nbody_data(csv_path::String, timestep_duration::Float64=1000.0)

Load N-body simulation data from CSV file.

# Arguments
- `csv_path`: Path to CSV file with columns: star, timestep, time, x, y, z
- `timestep_duration`: Duration of each timestep in years (default: 1000)

# Returns
- `NBodyData`: Structured data for position lookups
"""
function load_nbody_data(csv_path::String, timestep_duration::Float64=1000.0)
    println("Loading N-body data from: $csv_path")
    
    # Load CSV data
    df = CSV.read(csv_path, DataFrame)
    
    # Validate required columns
    required_cols = [:star, :timestep, :x, :y, :z]
    missing_cols = setdiff(required_cols, names(df))
    if !isempty(missing_cols)
        error("Missing required columns: $missing_cols")
    end
    
    # Build position dictionary
    positions = Dict{Tuple{Int,Int}, SVector{3,Float64}}()
    
    println("Processing $(nrow(df)) position records...")
    for row in eachrow(df)
        star_id = Int(row.star)
        timestep = Int(row.timestep)
        pos = SVector{3,Float64}(row.x, row.y, row.z)
        positions[(star_id, timestep)] = pos
    end
    
    # Extract metadata
    timesteps = unique(df.timestep)
    star_ids = unique(df.star)
    
    println("Loaded data for $(length(star_ids)) stars over $(length(timesteps)) timesteps")
    println("Timestep range: $(minimum(timesteps)) to $(maximum(timesteps))")
    
    return NBodyData(positions, timesteps, star_ids, timestep_duration)
end

"""
    get_position(nbody_data::NBodyData, star_id::Int, timestep::Int)

Get the position of a star at a specific timestep.

# Returns
- `SVector{3,Float64}`: Position vector, or throws error if not found
"""
function get_position(nbody_data::NBodyData, star_id::Int, timestep::Int)
    key = (star_id, timestep)
    if !haskey(nbody_data.positions, key)
        error("Position not found for star $star_id at timestep $timestep")
    end
    return nbody_data.positions[key]
end

"""
    has_position(nbody_data::NBodyData, star_id::Int, timestep::Int)

Check if position data exists for given star and timestep.
"""
function has_position(nbody_data::NBodyData, star_id::Int, timestep::Int)
    return haskey(nbody_data.positions, (star_id, timestep))
end

"""
    calculate_reachable_destinations(nbody_data::NBodyData, origin_star_id::Int, 
                                   current_timestep::Int, max_velocity::Float64,
                                   exclude_star_ids::Set{Int}=Set{Int}())

Calculate which stars can be reached from an origin star in one timestep.

# Arguments
- `nbody_data`: N-body trajectory data
- `origin_star_id`: ID of the launching star
- `current_timestep`: Current simulation timestep
- `max_velocity`: Maximum travel velocity in kpc/year
- `exclude_star_ids`: Set of star IDs to exclude (e.g., already living planets)

# Returns
- `Vector{Int}`: List of reachable destination star IDs
"""
function calculate_reachable_destinations(nbody_data::NBodyData, origin_star_id::Int, 
                                        current_timestep::Int, max_velocity::Float64,
                                        exclude_star_ids::Set{Int}=Set{Int}())
    # Check if we can launch (not at final timestep)
    arrival_timestep = current_timestep + 1
    if arrival_timestep > nbody_data.max_timestep
        return Int[]
    end
    
    # Get origin position at launch time
    origin_pos = get_position(nbody_data, origin_star_id, current_timestep)
    
    # Calculate maximum travel distance
    max_distance = max_velocity * nbody_data.timestep_duration  # kpc
    
    # Find reachable destinations
    reachable = Int[]
    for star_id in nbody_data.star_ids
        # Skip self and excluded stars
        if star_id == origin_star_id || star_id in exclude_star_ids
            continue
        end
        
        # Check if position data exists for arrival time
        if !has_position(nbody_data, star_id, arrival_timestep)
            continue
        end
        
        # Get destination position at arrival time
        dest_pos = get_position(nbody_data, star_id, arrival_timestep)
        
        # Check if reachable
        travel_distance = norm(dest_pos - origin_pos)
        if travel_distance <= max_distance
            push!(reachable, star_id)
        end
    end
    
    return reachable
end

"""
    select_most_similar_destination(origin_composition::Vector{<:Real}, 
                                   candidate_star_ids::Vector{Int},
                                   star_compositions::Dict{Int, Vector{<:Real}})

Select the most compositionally similar destination from candidates.

# Arguments
- `origin_composition`: Composition vector of the origin planet
- `candidate_star_ids`: List of candidate destination star IDs
- `star_compositions`: Mapping from star ID to composition vector

# Returns
- `Union{Int, Nothing}`: ID of most similar star, or nothing if no candidates
"""
function select_most_similar_destination(origin_composition::Vector{<:Real}, 
                                       candidate_star_ids::Vector{Int},
                                       star_compositions::Dict{Int, Vector{<:Real}})
    if isempty(candidate_star_ids)
        return nothing
    end
    
    # Calculate similarities (negative distance = higher similarity)
    similarities = Float64[]
    valid_candidates = Int[]
    
    for star_id in candidate_star_ids
        if haskey(star_compositions, star_id)
            similarity = -norm(origin_composition - star_compositions[star_id])
            push!(similarities, similarity)
            push!(valid_candidates, star_id)
        end
    end
    
    if isempty(similarities)
        return nothing
    end
    
    # Return star with highest similarity
    best_idx = argmax(similarities)
    return valid_candidates[best_idx]
end

"""
    plan_mission_nbody(origin_planet::Planet, current_timestep::Int, 
                      nbody_data::NBodyData, max_velocity::Float64,
                      living_star_ids::Set{Int}, claimed_star_ids::Set{Int},
                      star_compositions::Dict{Int, Vector{<:Real}})

Plan a mission from an origin planet using N-body data.

# Arguments
- `origin_planet`: Planet launching the mission
- `current_timestep`: Current simulation timestep
- `nbody_data`: N-body trajectory data
- `max_velocity`: Maximum travel velocity in kpc/year
- `living_star_ids`: Set of star IDs with living planets (to exclude)
- `claimed_star_ids`: Set of star IDs with claimed planets (to exclude)
- `star_compositions`: Mapping from star ID to composition vector

# Returns
- `Union{Int, Nothing}`: Destination star ID, or nothing if no valid destination
"""
function plan_mission_nbody(origin_planet, current_timestep::Int, 
                           nbody_data::NBodyData, max_velocity::Float64,
                           living_star_ids::Set{Int}, claimed_star_ids::Set{Int},
                           star_compositions::Dict{Int, Vector{<:Real}})
    
    # Find all stars to exclude (living + claimed + self)
    exclude_stars = union(living_star_ids, claimed_star_ids, Set([origin_planet.id]))
    
    # Calculate reachable destinations
    reachable_star_ids = calculate_reachable_destinations(
        nbody_data, origin_planet.id, current_timestep, max_velocity, exclude_stars
    )
    
    if isempty(reachable_star_ids)
        return nothing
    end
    
    # Select most similar destination
    destination_star_id = select_most_similar_destination(
        origin_planet.composition, reachable_star_ids, star_compositions
    )
    
    return destination_star_id
end

"""
Mission-based life spreading system for N-body integration
"""

# Mission structure for teleportation-based life spreading
Base.@kwdef mutable struct LifeMission{D} <: AbstractAgent
    id::Int
    departure_timestep::Int
    arrival_timestep::Int
    origin_star_id::Int
    destination_star_id::Int
    composition::Vector{<:Real}
    ancestors::Vector{<:AbstractAgent} = AbstractAgent[]
end

function Base.show(io::IO, mission::LifeMission{D}) where {D}
    s = "LifeMission with properties:"
    s *= "\n id: $(mission.id)"
    s *= "\n departure_timestep: $(mission.departure_timestep)"
    s *= "\n arrival_timestep: $(mission.arrival_timestep)"
    s *= "\n origin_star_id: $(mission.origin_star_id)"
    s *= "\n destination_star_id: $(mission.destination_star_id)"
    s *= "\n composition: $(mission.composition)"
    s *= "\n ancestors: $(length(mission.ancestors)) previous missions"
    print(io, s)
end

"""
    create_mission(origin_planet::Planet, destination_star_id::Int, 
                  current_timestep::Int, model_id_counter::Ref{Int})

Create a new life mission from origin to destination.

# Arguments
- `origin_planet`: Planet launching the mission
- `destination_star_id`: Target star ID
- `current_timestep`: Current simulation timestep
- `model_id_counter`: Reference to model's ID counter for unique mission IDs

# Returns
- `LifeMission`: New mission object
"""
function create_mission(origin_planet, destination_star_id::Int, 
                       current_timestep::Int, model_id_counter::Ref{Int})
    model_id_counter[] += 1
    
    mission = LifeMission{3}(
        id = model_id_counter[],
        departure_timestep = current_timestep,
        arrival_timestep = current_timestep + 1,  # Always 1 timestep travel
        origin_star_id = origin_planet.id,
        destination_star_id = destination_star_id,
        composition = copy(origin_planet.composition),
        ancestors = [] # TODO: implement ancestor tracking if needed
    )
    
    return mission
end

"""
    process_arriving_missions!(model, current_timestep::Int)

Process all missions arriving at the current timestep.

# Arguments
- `model`: ABM model containing missions and planets
- `current_timestep`: Current simulation timestep

# Returns
- `Int`: Number of successful terraforming events
"""
function process_arriving_missions!(model, current_timestep::Int)
    arriving_missions = filter(m -> m isa LifeMission && m.arrival_timestep == current_timestep, 
                              allagents(model))
    
    terraformed_count = 0
    
    for mission in arriving_missions
        # Find destination planet
        destination_planet = findfirst(p -> p isa Planet && p.id == mission.destination_star_id, 
                                     allagents(model))
        
        if destination_planet !== nothing
            dest_planet = model[destination_planet]
            
            # Check if destination is still available
            if !dest_planet.alive && !dest_planet.claimed
                # Terraform the destination
                terraform_from_mission!(dest_planet, mission, model)
                terraformed_count += 1
            end
        end
        
        # Remove the mission (whether successful or not)
        remove_agent!(mission, model)
    end
    
    return terraformed_count
end

"""
    terraform_from_mission!(planet::Planet, mission::LifeMission, model)

Terraform a planet based on an arriving mission.

# Arguments
- `planet`: Planet to terraform
- `mission`: Arriving life mission
- `model`: ABM model
"""
function terraform_from_mission!(planet::Planet, mission::LifeMission, model)
    # Mix compositions using model's composition mixing function
    if !isnothing(model.compmix_kwargs)
        planet.composition = model.compmix_func(mission.composition, planet.composition, model; model.compmix_kwargs...)
    else
        planet.composition = model.compmix_func(mission.composition, planet.composition, model)
    end
    
    # Update planet status
    planet.alive = true
    planet.claimed = true  # Keep as claimed to prevent multiple simultaneous arrivals
    
    # Track lineage (simplified version)
    push!(planet.parentcompositions, mission.composition)
    # Note: For full compatibility, we'd need to track parent planets and life agents
    
    # Update planet's last launch timestep to current time (ready to launch immediately)
    if hasfield(typeof(planet), :last_launch_timestep)
        planet.last_launch_timestep = model.current_timestep - 1  # Allow immediate launch
    end
end

"""
    attempt_launch_missions!(model, current_timestep::Int, max_velocity::Float64)

Attempt to launch new missions from all living planets.

# Arguments
- `model`: ABM model
- `current_timestep`: Current simulation timestep  
- `max_velocity`: Maximum travel velocity in kpc/year

# Returns
- `Int`: Number of missions launched
"""
function attempt_launch_missions!(model, current_timestep::Int, max_velocity::Float64)
    # Get all living planets
    living_planets = filter(p -> p isa Planet && p.alive, allagents(model))
    
    # Get sets of living and claimed star IDs
    living_star_ids = Set{Int}()
    claimed_star_ids = Set{Int}()
    
    for planet in filter(p -> p isa Planet, allagents(model))
        if planet.alive
            push!(living_star_ids, planet.id)
        end
        if planet.claimed
            push!(claimed_star_ids, planet.id)
        end
    end
    
    # Create star compositions mapping
    star_compositions = Dict{Int, Vector{<:Real}}()
    for planet in filter(p -> p isa Planet, allagents(model))
        star_compositions[planet.id] = planet.composition
    end
    
    missions_launched = 0
    
    for planet in living_planets
        # Check if planet can launch (respecting launch rate limit)
        if can_launch_mission(planet, current_timestep)
            # Plan mission using N-body data
            destination_star_id = plan_mission_nbody(
                planet, current_timestep, model.nbody_data, max_velocity,
                living_star_ids, claimed_star_ids, star_compositions
            )
            
            if destination_star_id !== nothing
                # Create and add mission
                mission = create_mission(planet, destination_star_id, current_timestep, 
                                       Ref(model.maxid))
                add_agent!(mission, model)
                
                # Update planet's launch tracking
                planet.last_launch_timestep = current_timestep
                
                # Mark destination as claimed
                dest_planet = findfirst(p -> p isa Planet && p.id == destination_star_id, 
                                      allagents(model))
                if dest_planet !== nothing
                    model[dest_planet].claimed = true
                end
                
                missions_launched += 1
            end
        end
    end
    
    return missions_launched
end

"""
    can_launch_mission(planet::Planet, current_timestep::Int)

Check if a planet can launch a mission at the current timestep.

# Arguments
- `planet`: Planet to check
- `current_timestep`: Current simulation timestep

# Returns
- `Bool`: True if planet can launch
"""
function can_launch_mission(planet::Planet, current_timestep::Int)
    if !planet.alive
        return false
    end
    
    # Check if planet has last_launch_timestep field
    if hasfield(typeof(planet), :last_launch_timestep)
        # Must wait at least 1 timestep between launches
        return (current_timestep - planet.last_launch_timestep) >= 1
    else
        # If no tracking field, allow launch (for backward compatibility)
        return true
    end
end

"""
Modified GalaxyParameters and model setup for N-body integration
"""

# Extended Planet type with N-body integration fields
Base.@kwdef mutable struct NBodyPlanet{D} <: AbstractAgent
    id::Int
    pos::SVector{D, Float64}
    vel::SVector{D, Float64} = SVector{D,Float64}(zeros(D))  # Not used in N-body mode
    
    composition::Vector{<:Real}
    initialcomposition::Vector{<:Real} = copy(composition)
    
    alive::Bool = false
    claimed::Bool = false
    spawn_threshold = 0  # Not used in mission-based system
    candidate_planets::Vector{<:AbstractAgent} = AbstractAgent[]  # Not used
    
    # Lineage tracking (simplified for N-body)
    parentplanets::Vector{<:AbstractAgent} = AbstractAgent[]
    parentlifes::Vector{<:AbstractAgent} = AbstractAgent[]
    parentcompositions::Vector{<:Vector{<:Real}} = Vector{Float64}[]
    
    # N-body specific fields
    last_launch_timestep::Int = -1000  # Allow immediate first launch
    reached_boundary::Bool = false
end

# Show method for NBodyPlanet
function Base.show(io::IO, planet::NBodyPlanet{D}) where {D}
    s = "NBodyPlanet in $(D)D space with properties:"
    s *= "\n id: $(planet.id)"
    s *= "\n pos: $(planet.pos)"
    s *= "\n composition: $(planet.composition)"
    s *= "\n alive: $(planet.alive)"
    s *= "\n claimed: $(planet.claimed)"
    s *= "\n last_launch_timestep: $(planet.last_launch_timestep)"
    print(io, s)
end

"""
    setup_nbody_model(nbody_data::NBodyData, initial_timestep::Int=0;
                     max_velocity::Float64=0.001,  # kpc/year
                     nool::Int=1,
                     ool::Union{Vector{Int}, Int, Nothing}=nothing,
                     compmix_func::Function=average_compositions,
                     compmix_kwargs=nothing,
                     compsize::Int=10,
                     maxcomp::Real=10.0)

Set up an ABM model using N-body data for planet positions.

# Arguments
- `nbody_data`: N-body trajectory data
- `initial_timestep`: Starting timestep for the simulation
- `max_velocity`: Maximum travel velocity in kpc/year
- `nool`: Number of planets to initialize with life
- `ool`: Specific planet IDs to initialize with life (if nothing, choose randomly)
- `compmix_func`: Function for mixing compositions during terraforming
- `compmix_kwargs`: Keyword arguments for composition mixing function
- `compsize`: Size of composition vectors
- `maxcomp`: Maximum value for composition elements

# Returns
- `ABM`: Initialized agent-based model
"""
function setup_nbody_model(nbody_data::NBodyData, initial_timestep::Int=0;
                          max_velocity::Float64=0.001,  # kpc/year  
                          nool::Int=1,
                          ool::Union{Vector{Int}, Int, Nothing}=nothing,
                          compmix_func::Function=average_compositions,
                          compmix_kwargs=nothing,
                          compsize::Int=10,
                          maxcomp::Real=10.0)
    
    # Determine space extent from initial positions
    initial_positions = [get_position(nbody_data, star_id, initial_timestep) 
                        for star_id in nbody_data.star_ids]
    
    # Calculate bounding box with some padding
    min_coords = minimum(hcat(initial_positions...), dims=2)[:, 1]
    max_coords = maximum(hcat(initial_positions...), dims=2)[:, 1]
    extent_size = max_coords - min_coords
    padding = 0.1 * maximum(extent_size)  # 10% padding
    
    extent = Tuple(extent_size .+ 2*padding)
    offset = min_coords .- padding
    
    # Create continuous space
    space = ContinuousSpace(extent; periodic=false)
    
    # Model properties
    properties = Dict(
        :current_timestep => initial_timestep,
        :nbody_data => nbody_data,
        :max_velocity => max_velocity,
        :dt => nbody_data.timestep_duration,  # Match N-body timestep
        :compsize => compsize,
        :maxcomp => maxcomp,
        :compmix_func => compmix_func,
        :compmix_kwargs => compmix_kwargs,
        :space_offset => offset,  # For coordinate transformation
        :n_living_planets => 0,
        :terraformed_on_step => false,
        :n_terraformed_on_step => 0,
        :missions_launched_on_step => 0
    )
    
    # Create model
    model = StandardABM(
        Union{NBodyPlanet, LifeMission},
        space;
        agent_step! = nbody_agent_step!,
        model_step! = nbody_model_step!,
        properties = properties,
        rng = Random.default_rng(),
        warn = false
    )
    
    # Add planets
    for star_id in nbody_data.star_ids
        # Get initial position and transform to model coordinates
        world_pos = get_position(nbody_data, star_id, initial_timestep)
        model_pos = SVector{3,Float64}(world_pos - offset)
        
        # Generate random composition
        composition = rand(model.rng, Uniform(0, maxcomp), compsize)
        
        # Create planet
        planet = NBodyPlanet{3}(
            id = star_id,  # Use star ID as planet ID
            pos = model_pos,
            composition = composition
        )
        
        add_agent_own_pos!(planet, model)
    end
    
    # Initialize living planets
    living_planet_ids = if ool === nothing
        # Choose random planets
        sample(nbody_data.star_ids, nool, replace=false)
    elseif isa(ool, Int)
        [ool]
    else
        ool
    end
    
    for planet_id in living_planet_ids
        planet = model[planet_id]
        planet.alive = true
        planet.claimed = true
        model.n_living_planets += 1
    end
    
    return model
end

"""
    nbody_agent_step!(agent, model)

Agent step function for N-body integrated model.
"""
function nbody_agent_step!(agent, model)
    if agent isa NBodyPlanet
        # Update planet position from N-body data
        world_pos = get_position(model.nbody_data, agent.id, model.current_timestep)
        model_pos = SVector{3,Float64}(world_pos - model.space_offset)
        agent.pos = model_pos
        
        # No other actions needed for planets in mission-based system
    elseif agent isa LifeMission
        # Missions are handled in model step, agents don't need individual steps
        # This function exists for compatibility but does nothing for missions
    end
end

"""
    nbody_model_step!(model)

Model step function for N-body integrated model.
"""
function nbody_model_step!(model)
    current_timestep = model.current_timestep
    
    # Process arriving missions first
    terraformed_count = process_arriving_missions!(model, current_timestep)
    
    # Update living planet count
    old_living_count = model.n_living_planets
    model.n_living_planets = count(p -> p isa NBodyPlanet && p.alive, allagents(model))
    
    # Track terraforming events
    model.terraformed_on_step = terraformed_count > 0
    model.n_terraformed_on_step = terraformed_count
    
    # Launch new missions
    missions_launched = attempt_launch_missions!(model, current_timestep, model.max_velocity)
    model.missions_launched_on_step = missions_launched
    
    # Advance to next timestep
    model.current_timestep += 1
    
    # Check if simulation should end
    if model.current_timestep > model.nbody_data.max_timestep
        println("Simulation ended: reached maximum timestep $(model.nbody_data.max_timestep)")
    end
end

"""
    run_nbody_simulation(nbody_data::NBodyData, n_steps::Int; kwargs...)

Run a complete N-body integrated terraforming simulation.

# Arguments
- `nbody_data`: N-body trajectory data
- `n_steps`: Number of timesteps to simulate
- `kwargs`: Additional arguments passed to setup_nbody_model

# Returns
- `ABM`: Final model state
- `Vector{Dict}`: Collected data from each step
"""
function run_nbody_simulation(nbody_data::NBodyData, n_steps::Int; kwargs...)
    # Setup model
    model = setup_nbody_model(nbody_data; kwargs...)
    
    # Data collection
    step_data = []
    
    println("Starting N-body terraforming simulation for $n_steps steps")
    
    for step in 1:n_steps
        # Record data before step
        step_info = Dict(
            :step => step,
            :timestep => model.current_timestep,
            :living_planets => model.n_living_planets,
            :total_planets => count(p -> p isa NBodyPlanet, allagents(model)),
            :active_missions => count(m -> m isa LifeMission, allagents(model))
        )
        
        # Step the model
        Agents.step!(model, 1)
        
        # Add post-step data
        step_info[:terraformed] = model.n_terraformed_on_step
        step_info[:missions_launched] = model.missions_launched_on_step
        
        push!(step_data, step_info)
        
        # Print progress
        if step % 100 == 0 || step <= 10
            println("Step $step: $(model.n_living_planets) living planets, $(step_info[:active_missions]) active missions")
        end
        
        # Stop if we've reached the end of N-body data
        if model.current_timestep > model.nbody_data.max_timestep
            println("Stopping: reached end of N-body data at timestep $(model.current_timestep)")
            break
        end
    end
    
    return model, step_data
end

# Export N-body integration functions
export NBodyPlanet, setup_nbody_model, nbody_agent_step!, nbody_model_step!, 
       run_nbody_simulation