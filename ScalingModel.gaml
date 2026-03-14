/**
* Name: ScalingModel
* Author: Gemini CLI
* Specifically for N=3000 and N=5000 Scaling Batch
*/

model ScalingModel

global {

    int num_days    <- 60;
    int current_day <- 0;

    // Batch control variables
    string scenario_id      <- "D1";
    int    repetition_id    <- 1;
    int    num_agents       <- 3000;

    // Experiment configuration subfolder
    string experiment_config <- "IC_ER";

    string input_folder  <- "C:/Users/moeez/Downloads/ATAI/projWeek/AI-Project/input_scaling/";
    string output_folder <- "C:/Users/moeez/Downloads/ATAI/projWeek/AI-Project/COMPR_output/";

    string run_tag      <- "001";
    string run_folder   <- "";
    string voters_file  <- "";
    string edges_file   <- "";
    string results_file <- "";

    string network_type     <- "erdos_renyi";
    string preference_model <- "IC";

    float prop_stubborn  <- 0.0;
    float prop_strategic <- 0.0;
    float prop_mixed     <- 0.0;

    list<string> candidates <- [
        "Macron", "Le Pen", "Melenchon", "Zemmour", "Pecresse",
        "Jadot", "Lassalle", "Roussel", "Dupont-Aignan",
        "Hidalgo", "Poutou", "Arthaud"
    ];

    map<string,int> candidate_scores <- map([]);
    list<string>    top2_candidates  <- [];

    float variance_scores <- 0.0;
    float social_welfare  <- 0.0;
    int   num_changes     <- 0;

    action reset_candidate_scores {
        candidate_scores <- map([]);
        loop c over: candidates {
            candidate_scores[c] <- 0;
        }
    }

    action compute_top2_global {
        list<string> sc <- candidates sort_by (- candidate_scores[each]);
        if length(sc) >= 2 {
            top2_candidates <- [sc[0], sc[1]];
        } else {
            top2_candidates <- copy(sc);
        }
    }

    action compute_global_metrics {
        do reset_candidate_scores;
        ask voter {
            if candidate_scores[current_vote] = nil {
                candidate_scores[current_vote] <- 0;
            }
            candidate_scores[current_vote] <- candidate_scores[current_vote] + 1;
        }
        do compute_top2_global;

        list<float> counts <- [];
        loop c over: candidates {
            counts <- counts + [float(candidate_scores[c])];
        }
        float m <- mean(counts);
        float s <- 0.0;
        loop val over: counts {
            s <- s + (val - m)^2;
        }
        variance_scores <- s / length(counts);
        num_changes <- length(voter where (each.changed_today));

        float welfare_sum <- 0.0;
        if length(top2_candidates) >= 1 {
            string cand1 <- top2_candidates[0];
            string cand2 <- cand1;
            if length(top2_candidates) >= 2 {
                cand2 <- top2_candidates[1];
            }
            ask voter {
                int w1 <- self.welfare_score_of(cand1);
                int w2 <- self.welfare_score_of(cand2);
                welfare_sum <- welfare_sum + min([w1, w2]);
            }
        }
        if length(voter) > 0 {
            social_welfare <- welfare_sum / length(voter);
        } else {
            social_welfare <- 0.0;
        }
    }

    action save_row {
        save [
            scenario_id, repetition_id, current_day, num_agents,
            network_type, preference_model,
            prop_stubborn, prop_strategic, prop_mixed,
            variance_scores, social_welfare, num_changes
        ]
        to:      results_file
        format:  "csv"
        header:  false
        rewrite: false;
    }

    init {
        if experiment_config = "IC_ER" {
            preference_model <- "IC";
            network_type     <- "erdos_renyi";
        } else if experiment_config = "IC_BA" {
            preference_model <- "IC";
            network_type     <- "barabasi_albert";
        } else if experiment_config = "Urn_ER" {
            preference_model <- "Urn";
            network_type     <- "erdos_renyi";
        } else if experiment_config = "Urn_BA" {
            preference_model <- "Urn";
            network_type     <- "barabasi_albert";
        }

        if scenario_id = "D1" { prop_stubborn <- 0.6; prop_strategic <- 0.2; prop_mixed <- 0.2; }
        else if scenario_id = "D2" { prop_stubborn <- 0.4; prop_strategic <- 0.3; prop_mixed <- 0.3; }
        else if scenario_id = "D3" { prop_stubborn <- 0.3; prop_strategic <- 0.4; prop_mixed <- 0.3; }
        else if scenario_id = "D4" { prop_stubborn <- 0.2; prop_strategic <- 0.6; prop_mixed <- 0.2; }
        else if scenario_id = "D5" { prop_stubborn <- 0.2; prop_strategic <- 0.3; prop_mixed <- 0.5; }

        run_tag <- "001"; 
        run_folder   <- input_folder + experiment_config + "/" + scenario_id + "_N" + string(num_agents) + "_run_" + run_tag + "/";
        voters_file  <- run_folder + "voters.csv";
        edges_file   <- run_folder + "edges.csv";
        results_file <- output_folder + "simulation_results_" + experiment_config + "_" + scenario_id + "_N" + string(num_agents) + "_run_" + run_tag + ".csv";

        matrix vm <- matrix(csv_file(voters_file, ",", true));
        loop i from: 0 to: vm.rows - 1 {
            create voter number: 1 {
                voter_id   <- int(vm[0, i]);
                agent_type <- string(vm[1, i]);
                loyalty    <- float(vm[2, i]);
                preferences <- [
                    string(vm[3,  i]), string(vm[4,  i]),
                    string(vm[5,  i]), string(vm[6,  i]),
                    string(vm[7,  i]), string(vm[8,  i]),
                    string(vm[9,  i]), string(vm[10, i]),
                    string(vm[11, i]), string(vm[12, i]),
                    string(vm[13, i]), string(vm[14, i])
                ];
                current_vote  <- string(vm[15, i]);
                previous_vote <- string(vm[15, i]);
                next_vote     <- string(vm[15, i]);
                changed_today <- false;
            }
        }

        num_agents <- length(voter);
        ask voter { my_neighbors <- []; }

        matrix em <- matrix(csv_file(edges_file, ",", true));
        loop i from: 0 to: em.rows - 1 {
            int s <- int(em[0, i]);
            int t <- int(em[1, i]);
            voter vs <- one_of(voter where (each.voter_id = s));
            voter vt <- one_of(voter where (each.voter_id = t));
            if vs != nil and vt != nil {
                if not (vt in vs.my_neighbors) { add vt to: vs.my_neighbors; }
                if not (vs in vt.my_neighbors) { add vs to: vt.my_neighbors; }
            }
        }

        do compute_global_metrics;
        save [
            "scenario_id","repetition_id","day","num_agents",
            "network_type","preference_model",
            "prop_stubborn","prop_strategic","prop_mixed",
            "variance_scores","social_welfare","num_changes"
        ]
        to:      results_file
        format:  "csv"
        header:  false
        rewrite: true;
        do save_row;
    }

    reflex step_simulation when: current_day < num_days {
        current_day <- current_day + 1;
        ask voter { previous_vote <- current_vote; }
        ask voter { do decide_next_vote; }
        ask voter {
            current_vote  <- next_vote;
            changed_today <- (current_vote != previous_vote);
        }
        do compute_global_metrics;
        do save_row;
    }
}

species voter {
    int voter_id;
    string agent_type;
    float loyalty;
    list<string> preferences;
    string current_vote;
    string previous_vote;
    string next_vote;
    bool changed_today <- false;
    list<voter>     my_neighbors <- [];
    map<string,int> local_poll   <- map([]);
    list<string>    local_top2   <- [];

    int rank_of (string cand) {
        int idx <- preferences index_of cand;
        if idx < 0 { return 12; }
        return idx + 1;
    }

    int welfare_score_of (string cand) {
        int idx <- preferences index_of cand;
        if idx < 0 { return 12; }
        return idx + 1;
    }

    string best_viable_candidate (list<string> viable) {
        if length(viable) = 0 { return preferences[0]; }
        string best <- viable[0];
        int best_rank <- rank_of(best);
        loop c over: viable {
            int r <- rank_of(c);
            if r < best_rank { best <- c; best_rank <- r; }
        }
        return best;
    }

    action update_local_poll {
        local_poll <- map([]);
        loop c over: candidates { local_poll[c] <- 0; }
        loop n over: my_neighbors {
            if local_poll[n.current_vote] = nil { local_poll[n.current_vote] <- 0; }
            local_poll[n.current_vote] <- local_poll[n.current_vote] + 1;
        }
        list<string> sc <- candidates sort_by (- local_poll[each]);
        if length(sc) >= 2 { local_top2 <- [sc[0], sc[1]]; } else { local_top2 <- copy(sc); }
    }

    action decide_next_vote {
        do update_local_poll;
        string favorite <- preferences[0];
        list<string> viable <- copy(local_top2);
        string best_v <- best_viable_candidate(viable);
        int total_votes <- length(my_neighbors);

        if agent_type = "stubborn" { next_vote <- favorite; }
        else if agent_type = "strategic" {
            if favorite in viable { next_vote <- favorite; } else { next_vote <- best_v; }
        } else {
            if favorite in viable { next_vote <- favorite; }
            else {
                float p_fav <- 0.0; float margin <- 1.0;
                if total_votes > 0 {
                    p_fav <- float(local_poll[favorite]) / total_votes;
                    if length(local_top2) >= 2 {
                        margin <- float(abs(local_poll[local_top2[0]] - local_poll[local_top2[1]])) / total_votes;
                    }
                }
                if (p_fav < 0.10) and (margin < 0.05) {
                    if flip(1.0 - loyalty) { next_vote <- best_v; } else { next_vote <- favorite; }
                } else { next_vote <- favorite; }
            }
        }
    }
}

experiment scaling_batch type: batch until: current_day = num_days {
    parameter "Config"          var: experiment_config among: ["IC_ER","IC_BA","Urn_ER","Urn_BA"];
    parameter "Scenario"        var: scenario_id       among: ["D1","D2","D3","D4","D5"];
    parameter "Population size" var: num_agents        among: [3000, 5000];
    parameter "Repetition ID"   var: repetition_id     among: [1];
}
