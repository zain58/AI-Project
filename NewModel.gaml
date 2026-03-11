/**
* Name: NewModel (Multi-Run Batch)
* Author: zainn
* Reads separate input folder for each run_id
*/

model NewModel1

global {

    int num_days    <- 60;
    int current_day <- 0;
    int run_id      <- 1;

   string input_folder  <- "C:/Users/moeez/Downloads/ATAI/GamaSetting/AIProject/input/";
   string output_folder <- "C:/Users/moeez/Downloads/ATAI/GamaSetting/AIProject/output/";

    string run_tag      <- "";
    string run_folder   <- "";
    string voters_file  <- "";
    string edges_file   <- "";
    string results_file <- "";

    string network_type     <- "erdos_renyi";
    string preference_model <- "IC";
    float prop_stubborn  <- 0.3;
    float prop_strategic <- 0.4;
    float prop_mixed     <- 0.3;

    int viability_delta <- 0;

    list<string> candidates <- [
        "Macron", "Le Pen", "Melenchon", "Zemmour", "Pecresse",
        "Jadot", "Lassalle", "Roussel", "Dupont-Aignan",
        "Hidalgo", "Poutou", "Arthaud"
    ];



    map<string,int> candidate_scores <- map([]);
    list<string> top2_candidates <- [];

    float variance_scores <- 0.0;
    float social_welfare  <- 0.0;
    int num_changes       <- 0;

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
                welfare_sum <- welfare_sum + max([w1, w2]);
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
            run_id, current_day, length(voter),
            network_type, preference_model,
            prop_stubborn, prop_strategic, prop_mixed,
            variance_scores, social_welfare, num_changes
        ]
        to: results_file
        format: "csv"
        header: false
        rewrite: false;
    }

    init {

        // Build padded run tag: 1 -> 001, 12 -> 012, 100 -> 100
        if run_id < 10 {
            run_tag <- "00" + string(run_id);
        } else if run_id < 100 {
            run_tag <- "0" + string(run_id);
        } else {
            run_tag <- string(run_id);
        }

        // Each run has its own input subfolder
        run_folder   <- input_folder + "run_" + run_tag + "/";
        voters_file  <- run_folder + "voters_run_" + run_tag + ".csv";
        edges_file   <- run_folder + "edges_run_" + run_tag + ".csv";
        results_file <- output_folder + "simulation_results_run_" + run_tag + ".csv";

        write "Starting run " + run_id + " using folder: " + run_folder;
        write "Results will be saved to: " + results_file;

        // Load voters
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

        ask voter {
            my_neighbors <- [];
        }

        // Load edges
        matrix em <- matrix(csv_file(edges_file, ",", true));
        loop i from: 0 to: em.rows - 1 {
            int s <- int(em[0, i]);
            int t <- int(em[1, i]);

            voter vs <- one_of(voter where (each.voter_id = s));
            voter vt <- one_of(voter where (each.voter_id = t));

            if vs != nil and vt != nil {
                if not (vt in vs.my_neighbors) {
                    add vt to: vs.my_neighbors;
                }
                if not (vs in vt.my_neighbors) {
                    add vs to: vt.my_neighbors;
                }
            }
        }

        do compute_global_metrics;

        // Write header row once
        save [
            "run_id","day","num_agents",
            "network_type","preference_model",
            "prop_stubborn","prop_strategic","prop_mixed",
            "variance_scores","social_welfare","num_changes"
        ]
        to: results_file
        format: "csv"
        header: false
        rewrite: true;

        do save_row;
    }

    reflex step_simulation when: current_day < num_days {

        current_day <- current_day + 1;

        ask voter {
            previous_vote <- current_vote;
        }

        ask voter {
            do decide_next_vote;
        }

        ask voter {
            current_vote  <- next_vote;
            changed_today <- (current_vote != previous_vote);
            if changed_today {
                switch_count     <- switch_count + 1;
                last_changed_day <- current_day;
            }
        }

        do compute_global_metrics;
        do save_row;
    }

    reflex end_run when: current_day = num_days {
        write "Run " + run_id + " complete. File saved: " + results_file;
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

    bool changed_today   <- false;
    int switch_count     <- 0;
    int last_changed_day <- -999;

    list<voter> my_neighbors <- [];
    map<string,int> local_poll <- map([]);
    list<string> local_top2 <- [];

    int rank_of (string cand) {
        int idx <- preferences index_of cand;
        if idx < 0 {
            return length(preferences);
        }
        return idx + 1;
    }

    int welfare_score_of (string cand) {
        int idx <- preferences index_of cand;
        if idx < 0 {
            return 0;
        }
        return length(preferences) - idx;
    }

    string best_viable_candidate (list<string> viable) {
        if length(viable) = 0 {
            return preferences[0];
        }
        string best <- viable[0];
        int best_rank <- rank_of(best);

        loop c over: viable {
            int r <- rank_of(c);
            if r < best_rank {
                best <- c;
                best_rank <- r;
            }
        }
        return best;
    }

    action update_local_poll {
        local_poll <- map([]);
        loop c over: candidates {
            local_poll[c] <- 0;
        }

        if length(my_neighbors) = 0 {
            local_poll[current_vote] <- 1;
        } else {
            loop n over: my_neighbors {
                if local_poll[n.current_vote] = nil {
                    local_poll[n.current_vote] <- 0;
                }
                local_poll[n.current_vote] <- local_poll[n.current_vote] + 1;
            }
        }

        list<string> sc <- candidates sort_by (- local_poll[each]);
        if length(sc) >= 2 {
            local_top2 <- [sc[0], sc[1]];
        } else {
            local_top2 <- copy(sc);
        }
    }

    action decide_next_vote {
        do update_local_poll;

        string favorite <- preferences[0];
        
        // Strategic assessment: who are the top 2 candidates in my local network?
        list<string> viable <- copy(local_top2);
        string best_v <- best_viable_candidate(viable);

        int total_votes <- 0;
        loop c over: candidates {
            total_votes <- total_votes + local_poll[c];
        }

        if agent_type = "stubborn" {
            next_vote <- favorite;

        } else if agent_type = "strategic" {
            // Model 2: Pure Strategic Agent (Expected Utility Maximizer)
            // Logic: EU = P(win) * U(win). 
            // This agent treats P(win) as zero if their favorite is not in the top-2 (viable).
            // If EU(favorite) is zero, they switch to the top candidate in their preferences 
            // that has a mathematical chance (P(win) > 0) by being in the top-2.
            if favorite in viable {
                next_vote <- favorite;
            } else {
                next_vote <- best_v;
            }

        } else { // Model 3: Mixed / In-Between Agent (Bounded Rationality)
            float p_fav <- 0.0;
            float margin <- 1.0; // Default to wide margin
            
            if total_votes > 0 {
                p_fav <- float(local_poll[favorite]) / total_votes;
                
                if length(local_top2) >= 2 {
                    margin <- float(local_poll[local_top2[0]] - local_poll[local_top2[1]]) / total_votes;
                }
            }

            // Tipping Point Logic (Two-Factor Rule):
            // 1. Abandonment: P(favorite) < 10% 
            // 2. Efficacy: Race is tight (Margin < 5%)
            if (p_fav < 0.10) and (margin < 0.05) {
                // Risk penalty logic: higher loyalty = lower chance to take the risk of switching
                if flip(1.0 - loyalty) {
                    next_vote <- best_v;
                } else {
                    next_vote <- favorite;
                }
            } else {
                // Stay loyal if candidate is still viable (>10%) or race isn't tight (>5%)
                next_vote <- favorite;
            }
        }
    }
}

// GUI experiment — single run
experiment main type: gui {
    parameter "Run ID" var: run_id min: 1 max: 999;

    output {
        monitor "Day" value: current_day;
        monitor "Variance of scores" value: variance_scores;
        monitor "Social welfare" value: social_welfare;
        monitor "Number of opinion changes" value: num_changes;

        display "Daily indicators" type: 2d {
            chart "Daily indicators" type: series {
                data "Variance" value: float(variance_scores) color: #blue;
                data "Social welfare" value: float(social_welfare) color: #green;
                data "Opinion changes" value: float(num_changes) color: #red;
            }
        }
    }
}

// Batch experiment — automatic 50 runs
experiment batch_runs type: batch until: current_day = num_days {
    parameter "Run ID" var: run_id among: [1,2,3,4,5,6,7,8,9,10,
                                           11,12,13,14,15,16,17,18,19,20,
                                           21,22,23,24,25,26,27,28,29,30,
                                           31,32,33,34,35,36,37,38,39,40,
                                           41,42,43,44,45,46,47,48,49,50];
}
