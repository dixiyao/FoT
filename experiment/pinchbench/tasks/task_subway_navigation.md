---
id: task_subway_navigation
name: NYC Subway Navigation
category: productivity
grading_type: llm_judge
timeout_seconds: 180
workspace_files:
  - path: "subway_map.md"
    content: |
      # NYC Subway — Simplified Route Map

      This is a simplified representation of key NYC subway lines and their major stops.
      Use this to plan routes between stations.

      ## Line 1 (Red — Local)
      Van Cortlandt Park-242 St → 238 St → 231 St → Marble Hill-225 St → 215 St → 207 St → Dyckman St → 191 St → 181 St → 168 St → 157 St → 145 St → 137 St-City College → 125 St → 116 St-Columbia University → Cathedral Pkwy-110 St → 103 St → 96 St → 86 St → 79 St → 72 St → 66 St-Lincoln Center → 59 St-Columbus Circle → 50 St → Times Sq-42 St → 34 St-Penn Station → 28 St → 23 St → 18 St → 14 St-7th Ave → Christopher St-Sheridan Sq → Houston St → Canal St → Franklin St → Chambers St → Cortlandt St-WTC → Rector St → South Ferry

      ## Line 2 (Red — Express)
      Wakefield-241 St → Nereid Ave → 233 St → 225 St → 219 St → Gun Hill Rd → Burke Ave → Allerton Ave → Pelham Pkwy → Bronx Park East → E 180 St → West Farms Sq-E Tremont Ave → 174 St → Freeman St → Simpson St → Intervale Ave → Prospect Ave → Jackson Ave-Westchester Ave → 149 St-Grand Concourse → 135 St → 125 St → 116 St → Central Park North-110 St → **96 St** → **72 St** → **Times Sq-42 St** → **34 St-Penn Station** → **14 St-7th Ave** → **Chambers St** → Fulton St → Wall St → Clark St → Borough Hall → Hoyt St → Nevins St → Atlantic Ave-Barclays Ctr → Bergen St → Grand Army Plaza → Eastern Pkwy-Brooklyn Museum → Franklin Ave-Medgar Evers College → President St-Medgar Evers College → Sterling St → Winthrop St → Church Ave → Beverly Rd → Newkirk Ave-Little Haiti → Flatbush Ave-Brooklyn College

      ## Line 4 (Green — Express)
      Woodlawn → Mosholu Pkwy → Bedford Park Blvd → Kingsbridge Rd → Fordham Rd → 183 St → Burnside Ave → 176 St → Mt Eden Ave → 170 St → 167 St-Yankee Stadium → 161 St-Yankee Stadium → 149 St-Grand Concourse → 125 St → **86 St** → **59 St** → Grand Central-42 St → **14 St-Union Sq** → Brooklyn Bridge-City Hall → Fulton St → Wall St → Borough Hall → Atlantic Ave-Barclays Ctr → Nevins St → → Crown Hts-Utica Ave

      ## Line 7 (Purple)
      Flushing-Main St → Mets-Willets Point → Junction Blvd → 90 St-Elmhurst Ave → 82 St-Jackson Hts → 74 St-Broadway → 69 St → Woodside-61 St → 52 St → 46 St-Bliss → 40 St-Lowery → 33 St-Rawson → Queensboro Plaza → Court Sq → Hunters Point Ave → Vernon Blvd-Jackson Ave → Grand Central-42 St → Times Sq-42 St → 34 St-Hudson Yards

      ## Line A (Blue — Express)
      Inwood-207 St → Dyckman St → 190 St → 181 St → 175 St → 168 St → 145 St → **125 St** → **59 St-Columbus Circle** → **42 St-Port Authority** → **34 St-Penn Station** → **14 St-8th Ave** → West 4 St-Washington Sq → Spring St → Canal St → Fulton St → High St → Jay St-MetroTech → Hoyt-Schermerhorn → Lafayette Ave → Clinton-Washington Aves → → Far Rockaway / Lefferts Blvd / Rockaway Park

      ## Line L (Gray)
      8 Ave-14 St → 6 Ave-14 St → Union Sq-14 St → 3 Ave → 1 Ave → Bedford Ave → Lorimer St → Graham Ave → Grand St → Montrose Ave → Morgan Ave → Jefferson St → DeKalb Ave → Myrtle-Wyckoff → Halsey St → Wilson Ave → Bushwick Ave → Broadway Junction → Atlantic Ave → Sutter Ave → Livonia Ave → New Lots Ave → East 105 St → Canarsie-Rockaway Pkwy

      ## Transfer Stations (key connections)
      - **Times Sq-42 St**: 1, 2, 3, 7, N, Q, R, W, S, A, C, E
      - **14 St-Union Sq**: 4, 5, 6, L, N, Q, R, W
      - **59 St-Columbus Circle**: 1, 2, A, B, C, D
      - **Grand Central-42 St**: 4, 5, 6, 7, S
      - **Fulton St**: 2, 3, 4, 5, A, C, J, Z
      - **Atlantic Ave-Barclays Ctr**: 2, 3, 4, 5, B, D, N, Q, R
      - **14 St-7th Ave / 14 St-8th Ave**: Connected via walkway (1, 2, 3 ↔ A, C, E, L)
      - **34 St-Penn Station**: 1, 2, 3, A, C, E
      - **Jay St-MetroTech**: A, C, F, R
      - **125 St**: 1 (Broadway), 2/3 (Lenox), A/B/C/D (St. Nicholas), 4/5/6 (Lexington — separate station at 125 St)
      - **168 St**: 1, A, C
---

## Prompt

Using the subway map provided in `subway_map.md`, give me directions for the following route:

**From:** Bedford Ave station (L train, Williamsburg, Brooklyn)
**To:** Yankee Stadium (161 St-Yankee Stadium, 4 train, Bronx)

Provide:

1. **Step-by-step directions** with specific train lines and transfer stations
2. **Estimated number of stops** for each leg
3. **Total estimated travel time** (assume 2-3 minutes per stop, 5-10 minutes per transfer)
4. **Alternative route** if one exists
5. **Tips** — best car position for transfers, rush hour considerations

Save your directions to `subway_directions.md`.

## Expected Behavior

The agent should:

1. Read the subway map file
2. Plan a route from Bedford Ave (L) to 161 St-Yankee Stadium (4)
3. Identify transfer points (likely Union Sq L→4, or 14 St to connect to a line reaching Yankee Stadium)
4. Calculate approximate stop counts and travel time
5. Consider alternative routes
6. Save clear directions to `subway_directions.md`

The most logical route involves:
- L train from Bedford Ave to Union Sq-14 St (3 stops)
- Transfer to 4 train (express) at 14 St-Union Sq
- 4 train uptown to 161 St-Yankee Stadium

An alternative might involve:
- L train to 14 St, walk to 14 St-7th Ave, take 2 express uptown (but this doesn't directly reach Yankee Stadium without another transfer)

## Grading Criteria

- [ ] File `subway_directions.md` created
- [ ] Route uses L train from Bedford Ave
- [ ] Identifies Union Sq or 14 St as transfer point
- [ ] Routes to 161 St-Yankee Stadium
- [ ] Uses 4 train for the Bronx leg
- [ ] Stop counts provided
- [ ] Travel time estimated
- [ ] Alternative route suggested
- [ ] Directions are clear and followable

## LLM Judge Rubric

### Criterion 1: Route Correctness (Weight: 35%)

**Score 1.0**: Primary route is correct and optimal — L from Bedford Ave to Union Sq, transfer to 4 express uptown to 161 St-Yankee Stadium. Transfer stations are accurately identified. No impossible or fictional connections.
**Score 0.75**: Route is correct but may include a suboptimal transfer or minor station name error.
**Score 0.5**: Route gets from A to B but includes an unnecessary transfer or uses a significantly suboptimal path.
**Score 0.25**: Route has major errors (wrong direction, impossible transfer) but shows some understanding.
**Score 0.0**: Route is completely wrong or missing.

### Criterion 2: Practical Detail (Weight: 25%)

**Score 1.0**: Includes stop counts for each leg, realistic time estimates (total ~40-50 minutes), transfer walking directions ("cross the platform" vs. "follow signs"), and rush hour tips. Information would be useful to someone actually making this trip.
**Score 0.75**: Good practical detail with minor gaps. Time estimates reasonable.
**Score 0.5**: Basic directions but lacking detail a traveler would want.
**Score 0.25**: Minimal practical information.
**Score 0.0**: No practical detail.

### Criterion 3: Alternative Route (Weight: 20%)

**Score 1.0**: Provides a legitimate alternative route with reasoning for when it might be better (e.g., if 4 train is delayed, service changes on weekends). Alternative is actually viable per the map data.
**Score 0.75**: Provides an alternative that works but reasoning is thin.
**Score 0.5**: Alternative mentioned but may not be optimal or is barely different.
**Score 0.25**: Vague mention of alternatives without specifics.
**Score 0.0**: No alternative route.

### Criterion 4: Map Interpretation (Weight: 20%)

**Score 1.0**: Demonstrates accurate reading of the provided subway map. Uses correct station names, correctly identifies express vs. local stops (bolded in map = express), and respects transfer connections listed in the map.
**Score 0.75**: Mostly accurate map reading with minor errors.
**Score 0.5**: Shows some map reading but with notable errors or confusion.
**Score 0.25**: Minimal evidence of actually reading the provided map.
**Score 0.0**: Ignores the map entirely or misreads it significantly.

## Additional Notes

- A text-based subway map is provided instead of an image to make this task accessible to all agents. The map includes enough detail to plan routes.
- Bold station names in the Line 2 and Line 4 descriptions indicate express stops.
- The Bedford Ave → Yankee Stadium route was chosen because it requires exactly one transfer and has a clear optimal path, making it easy to grade.
- The transfer at Union Sq (L → 4) is one of the most well-known connections in the NYC subway system.
- Rush hour and service change tips test whether the agent goes beyond basic routing to provide genuinely helpful travel advice.
