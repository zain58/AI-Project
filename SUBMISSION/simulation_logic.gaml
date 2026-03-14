model NewModel

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

    list<string> candidates <- [
        "Macron", "Le Pen", "Melenchon", "Zemmour", "Pecresse",
        "Jadot", "Lassalle", "Roussel", "Dupont-Aignan",
        "Hidalgo", "Poutou", "Arthaud"
    ];

    map<string,int> scores <- map([]);
    list<string> finalists <- [];

    float variance       <- 0.0;
    float welfare        <- 0.0;
    int   changes_today  <- 0;

    action reset_scores {
        scores <- map([]);
        loop c over: candidates { scores[c] <- 0; }
    }

    action find_finalists {
        list<string> sorted_c <- candidates sort_by (- scores[each]);
        if length(sorted_c) >= 2 {
            finalists <- [sorted_c[0], sorted_c[1]];
        } else {
            finalists <- copy(sorted_c);
        }
    }

    action calc_metrics {
        do reset_scores;
        ask voter {
            if scores[current_vote] = nil { scores[current_vote] <- 0; }
            scores[current_vote] <- scores[current_vote] + 1;
        }
        do find_finalists;

        list<float> counts <- [];
        loop c over: candidates { counts <- counts + [float(scores[c])]; }

        float avg_val <- mean(counts);
        float sum_sq  <- 0.0;
        loop v over: counts { sum_sq <- sum_sq + (v - avg_val)^2; }
        variance <- sum_sq / length(counts);

        changes_today <- length(voter where (each.changed));

        float w_sum <- 0.0;
        if length(finalists) >= 1 {
            string c1 <- finalists[0];
            string c2 <- length(finalists) >= 2 ? finalists[1] : c1;
            ask voter {
                int pts1 <- self.get_welfare(c1);
                int pts2 <- self.get_welfare(c2);
                if (c1 != c2) { w_sum <- w_sum + pts1 + pts2; } else { w_sum <- w_sum + pts1; }
            }
        }
        welfare <- length(voter) > 0 ? w_sum / length(voter) : 0.0;
    }

    action log_step {
        save [run_id, current_day, variance, welfare, changes_today] to: results_file format: "csv" rewrite: false;
    }

    init {
        run_tag <- (run_id < 10 ? "00" : (run_id < 100 ? "0" : "")) + string(run_id);
        run_folder   <- input_folder + "run_" + run_tag + "/";
        voters_file  <- run_folder + "voters_run_" + run_tag + ".csv";
        edges_file   <- run_folder + "edges_run_" + run_tag + ".csv";
        results_file <- output_folder + "simulation_results_run_" + run_tag + ".csv";

        matrix data_v <- matrix(csv_file(voters_file, ",", true));
        loop i from: 0 to: data_v.rows - 1 {
            create voter {
                voter_id   <- int(data_v[0, i]);
                type       <- string(data_v[1, i]);
                loyalty    <- float(data_v[2, i]);
                prefs      <- [string(data_v[3,i]), string(data_v[4,i]), string(data_v[5,i]), string(data_v[6,i]),
                               string(data_v[7,i]), string(data_v[8,i]), string(data_v[9,i]), string(data_v[10,i]),
                               string(data_v[11,i]), string(data_v[12,i]), string(data_v[13,i]), string(data_v[14,i])];
                current_vote <- string(data_v[15, i]);
                old_vote     <- current_vote;
                next_vote    <- current_vote;
                changed      <- false;
            }
        }

        matrix data_e <- matrix(csv_file(edges_file, ",", true));
        loop i from: 0 to: data_e.rows - 1 {
            voter s <- one_of(voter where (each.voter_id = int(data_e[0, i])));
            voter t <- one_of(voter where (each.voter_id = int(data_e[1, i])));
            if s != nil and t != nil {
                if !(t in s.neighbors) { add t to: s.neighbors; }
                if !(s in t.neighbors) { add s to: t.neighbors; }
            }
        }

        do calc_metrics;
        save ["run_id","day","variance_scores","social_welfare","num_changes"] to: results_file format: "csv" rewrite: true;
        do log_step;
    }

    reflex simulate when: current_day < num_days {
        current_day <- current_day + 1;
        ask voter { old_vote <- current_vote; }
        ask voter { do decide; }
        ask voter {
            current_vote <- next_vote;
            changed      <- (current_vote != old_vote);
        }
        do calc_metrics;
        do log_step;
    }
}

species voter {
    int voter_id;
    string type;
    float loyalty;
    list<string> prefs;
    string current_vote;
    string old_vote;
    string next_vote;
    bool changed <- false;
    list<voter> neighbors <- [];
    map<string,int> local_results <- map([]);
    list<string> local_top2 <- [];

    int get_rank(string c) {
        int i <- prefs index_of c;
        return i < 0 ? 12 : i + 1;
    }

    int get_welfare(string c) {
        int i <- prefs index_of c;
        return i < 0 ? 12 : i + 1;
    }

    string get_best_viable(list<string> viable) {
        if length(viable) = 0 { return prefs[0]; }
        string b <- viable[0];
        int r_best <- get_rank(b);
        loop v over: viable {
            int r <- get_rank(v);
            if r < r_best { b <- v; r_best <- r; }
        }
        return b;
    }

    action update_local {
        local_results <- map([]);
        loop c over: candidates { local_results[c] <- 0; }
        loop n over: neighbors {
            local_results[n.current_vote] <- local_results[n.current_vote] + 1;
        }
        list<string> sc <- candidates sort_by (- local_results[each]);
        local_top2 <- length(sc) >= 2 ? [sc[0], sc[1]] : copy(sc);
    }

    action decide {
        do update_local;
        string fav <- prefs[0];
        list<string> viable <- copy(local_top2);
        string best_v <- get_best_viable(viable);
        int total <- length(neighbors);

        if type = "stubborn" { next_vote <- fav; }
        else if type = "strategic" {
            next_vote <- fav in viable ? fav : best_v;
        } else {
            if fav in viable { next_vote <- fav; }
            else {
                float p_fav <- total > 0 ? float(local_results[fav]) / total : 0.0;
                float gap <- (total > 0 and length(local_top2) >= 2) ? float(abs(local_results[local_top2[0]] - local_results[local_top2[1]])) / total : 1.0;
                if (p_fav < 0.10) and (gap < 0.05) {
                    next_vote <- flip(1.0 - loyalty) ? best_v : fav;
                } else { next_vote <- fav; }
            }
        }
    }
}

experiment main type: gui until: (current_day = num_days) {
    parameter "Run ID" var: run_id;
    output {
        monitor "Day" value: current_day;
        display "Charts" {
            chart "Dynamics" type: series {
                data "Var" value: float(variance) color: #blue;
                data "Welfare" value: float(welfare) color: #green;
                data "Changes" value: float(changes_today) color: #red;
            }
        }
    }
}

experiment batch_runs type: batch until: current_day = num_days {
    parameter "Run ID" var: run_id among: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50];
}
