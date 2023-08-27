extensions [ py ]

globals [
  selected-car        ; the currently selected car
  car-number
  lanes               ; a list of the y coordinates of different lanes
  speed-max
  speed-min
  speed-sudden-stop   ; the speed that a car has when its car ahead is damaged
  observation-max     ; the maximum distance that a car can observe
  damaged-rate        ; the rate at which a car is damaged when traveling on a road
  damaged-color

  change-lane-fd      ; the value that we forward a car when it changes lane
  blocking-car-observation ; the distance that we use to observe blocking cars

  ; parameters for reinforcement learning
  inputs
  ac-update-steps
]

turtles-own [
  speed         ; the current speed of the car
  speed-top     ; the maximum speed of the car (different for all cars)
  target-lane   ; the desired lane of the car
  patience      ; the driver's current level of patience
  init-xcor     ; the initial x coordinate for REINFORCE algo
  id

  ; attributes for reinforcement learning
  reward
  state
  action        ; 0 = decelerate, 1 = stay same, 2 = accelerate, 3 = change lane
  next-state
]

to setup
  clear-all

  set-default-shape turtles "car"
  set car-number 0
  set speed-max 1   ; = 100 km/h
  set speed-min 0   ; = 0 km/h
  set speed-sudden-stop 0.05 ; = 5 km/h
  set observation-max world-width / 2
  set damaged-rate 0.0001 ; from 0 to 1
  set damaged-color orange
  set change-lane-fd 0.10
  set blocking-car-observation 1.25
  set ac-update-steps 2000

  draw-road
  create-or-remove-cars

  set selected-car one-of turtles
  ask selected-car [ set color pink ]

  if move-strategy != "Greedy" [  ; parameters if we choose a RL algorithms
    py:setup py:python
    py:run "import numpy as np"

    py:set "action_size" 4   ; 0 = decelerate, 1 = stay same, 2 = accelerate, 3 = change lane
    py:set "gamma" discount-factor
    py:set "alpha" learning-rate
  ]

  if (move-strategy = "Deep Q-Learning") or (move-strategy = "Deep Q-Network") [
    setup-dql-dqn
  ]

  if (move-strategy = "Naive Actor-Critic") or (move-strategy = "Advantage Actor-Critic") [
    setup-ac
  ]

  if (move-strategy = "Reinforce")  [
    setup-reinforce
  ]

  reset-ticks
end

to go
  if ticks >= simulation-time [stop]

  ;if (random-float 1 > 0.95) [
  ;  set number-of-cars number-of-cars + random 2
  ;]

  ;create-or-remove-cars

  if move-strategy = "Greedy" [
    ask turtles with [speed > 0]
    [
      move-forward-greedy
    ]
  ]

  if (move-strategy = "Deep Q-Learning") or (move-strategy = "Deep Q-Network") [
    go-dql-dqn
  ]

  if (move-strategy = "Naive Actor-Critic") [
    go-nac
  ]

  if (move-strategy = "Advantage Actor-Critic") [
    go-a2c
  ]

  if (move-strategy = "Reinforce") [
    go-reinforce
  ]

  tick
end

;; setup for Deep Q-Learning and Deep Q-Network
;; in fact, DQL is the a version of DQN without Target Network
to setup-dql-dqn
  set inputs (list
    [-> speed]
    [-> get-distance-to-car get-car-ahead-same-lane]
    [-> get-speed-to-car get-car-ahead-same-lane]
    [-> get-distance-to-car get-car-behind-same-lane]
    [-> get-speed-to-car get-car-behind-same-lane])

  if input-exp? [
    set inputs lput [-> get-epsilon ] inputs
  ]

  if input-time? [
    set inputs lput [-> ticks] inputs
  ]

  py:set "input_size" length inputs
  py:set "hl_size" 36
  py:set "memory_size" memory-size
  py:set "batch_size" batch-size

  (py:run
    "from tensorflow import keras"
    "from keras import layers"
    "Q_network = keras.Sequential()"
    "Q_network.add(layers.Dense(hl_size, input_shape=(input_size,), activation='relu'))"
    "Q_network.add(layers.Dense(hl_size, activation='relu'))"
    "Q_network.add(layers.Dense(hl_size, activation='relu'))"
    "Q_network.add(layers.Dense(action_size))"
    "optimizer = keras.optimizers.Adam(learning_rate=alpha)"
    "Q_network.compile(optimizer, 'mse')"
    ;"Q_network.summary()"
    "memory = []")

  if move-strategy = "Deep Q-Network" [
    py:set "update_steps" 1000    ;; hyperparameter to update target (before: c = 5)
    py:set "step" 0

    (py:run "Q_hat_network = keras.models.clone_model(Q_network)")
  ]

end

;; setup for actor-critic algorithms (Naive AC, A2C without multiple workers, A2C with multiple workers)
;; use 2 networks: one for Actor and one for Critic
;; Actor network is used to select action
;; Critic network is used to calculate state value
;; A policy function (or policy) returns a probability distribution over actions that the agent can take based on the given state
to setup-ac
  set inputs (list
    [-> speed]
    [-> get-distance-to-car get-car-ahead-same-lane]
    [-> get-speed-to-car get-car-ahead-same-lane]
    [-> get-distance-to-car get-car-behind-same-lane]
    [-> get-speed-to-car get-car-behind-same-lane])

  if input-exp? [
    set inputs lput [-> get-epsilon ] inputs
  ]

  if input-time? [
    set inputs lput [-> ticks] inputs
  ]

  py:set "input_size" length inputs
  py:set "hl_size" 36

  (py:run
    "import tensorflow as tf"
    "import tensorflow_probability as tfp"
    "from tensorflow import keras"
    "from keras import layers"
    "import tensorflow.keras.losses as kls"
    ;; define Actor network
    "Actor_network = keras.Sequential()"
    "Actor_network.add(layers.Dense(hl_size, input_shape=(input_size,), activation='relu'))"
    "Actor_network.add(layers.Dense(hl_size, activation='relu'))"
    "Actor_network.add(layers.Dense(hl_size, activation='relu'))"
    "Actor_network.add(layers.Dense(action_size, activation='softmax'))"
    "a_optimizer = keras.optimizers.Adam(learning_rate=alpha)"
    ;"Actor_network.compile(a_optimizer, 'mse')"
    ;"Actor_network.summary()"
    ;; define Critic network
    "Critic_network = keras.Sequential()"
    "Critic_network.add(layers.Dense(hl_size, input_shape=(input_size,), activation='relu'))"
    "Critic_network.add(layers.Dense(hl_size, activation='relu'))"
    "Critic_network.add(layers.Dense(hl_size, activation='relu'))"
    "Critic_network.add(layers.Dense(1))"
    "c_optimizer = keras.optimizers.Adam(learning_rate=alpha)"
    ;"Critic_network.compile(c_optimizer, 'mse')"
    ;"Critic_network.summary()"
  )

  if move-strategy = "Advantage Actor-Critic" [
    py:set "update_steps" ac-update-steps
    py:set "update_offset" (round (ac-update-steps / number-of-cars) )
    py:set "step" 0
    py:set "nb_cars" ((max [ who ] of turtles) + 1)

    ; 4 matrices to store state, action, reward, return
    (py:run
      "mem_states = [[] for i in range(nb_cars)]"
      "mem_rewards = [[] for i in range(nb_cars)]"
      "mem_actions = [[] for i in range(nb_cars)]"
      "mem_returns = [[] for i in range(nb_cars)]")
  ]
end

;;setup reinforce algorithm: Monte-Carlo Policy-Gradient Control
to setup-reinforce
  set inputs (list
    [-> speed]
    [-> get-distance-to-car get-car-ahead-same-lane]
    [-> get-speed-to-car get-car-ahead-same-lane]
    [-> get-distance-to-car get-car-behind-same-lane]
    [-> get-speed-to-car get-car-behind-same-lane])

  if input-exp? [
    set inputs lput [-> get-epsilon ] inputs
  ]

  if input-time? [
    set inputs lput [-> ticks] inputs
  ]

  py:set "input_size" length inputs
  py:set "hl_size" 36
  (py:run
    "import tensorflow as tf"
    "import numpy as np"
    "import tensorflow_probability as tfp"
    "from tensorflow import keras"
    "from tensorflow.keras import layers"
    "R_network = keras.Sequential()"
    "R_network.add(layers.Dense(hl_size, input_shape=(input_size,), activation='relu'))"
    "R_network.add(layers.Dense(hl_size, activation='relu'))"
    "R_network.add(layers.Dense(hl_size, activation='relu'))"
    "R_network.add(layers.Dense(action_size, activation='softmax'))"
    "optimizer = keras.optimizers.Adam(learning_rate=alpha)")

  py:set "nb_cars" car-number + 1 ;((max [ who ] of turtles) + 1)
  ; 4 matrices to store state, action, reward
  (py:run
    "mem_states = [[] for i in range(nb_cars)]"
    "mem_rewards = [[] for i in range(nb_cars)]"
    "mem_actions = [[] for i in range(nb_cars)]"
    "mem_returns = [[] for i in range(nb_cars)]")
end

;; go for Deep Q-Learning and Deep Q-Network
to go-dql-dqn
  ; get current states
  ask turtles [
    set state map runresult inputs
  ]
  let turtle-list sort turtles
  py:set "states" map [ t -> [ state ] of t ] turtle-list

  ; agents choose actions, using e-greedy
  let actions py:runresult "np.argmax(Q_network.predict(np.array(states)), axis = 1)"
  (foreach turtle-list actions [ [t a] ->
    ask t [
      ifelse (0.05 + random-float 1) < get-epsilon [
        set action random 4
        while [action = 3 and patience > 0] [set action random 4]
      ] [
        set action a
      ]
    ]
  ])

  ; perform chosen actions, i.e. forward cars and get next state
  ask turtles [
    move-forward-rl
    set next-state map runresult inputs
  ]

  ; update Experience Replay to train Q-network
  let data [ (list state action reward next-state) ] of turtles
  py:set "new_experience" data
  (py:run
    "memory.extend(new_experience)"
    "if len(memory) > memory_size:"
    "    memory = memory[-memory_size:]"
    "sample_ix = np.random.randint(len(memory), size = batch_size)"
    "states = np.array([memory[i][0] for i in sample_ix])"
    "actions = np.array([memory[i][1] for i in sample_ix])"
    "rewards = np.array([memory[i][2] for i in sample_ix])"
    "next_states = np.array([memory[i][3] for i in sample_ix])"
    "targets = Q_network.predict(states)" )

  if move-strategy = "Deep Q-Learning" [
    (py:run "next_rewards = Q_network.predict(next_states)")
  ]

  if move-strategy = "Deep Q-Network" [
    (py:run "next_rewards = Q_hat_network.predict(next_states)")
  ]

  (py:run
    "targets[np.arange(targets.shape[0]), actions] = rewards + gamma*np.max(next_rewards, axis = 1)"
    "Q_network.train_on_batch(states, targets)")

  if move-strategy = "Deep Q-Network" [
    ; update periodically target network
    (py:run
      "step = step + 1"
      "if (step % update_steps == 0): Q_hat_network = keras.models.clone_model(Q_network)")
  ]
end

;; go for Naive Actor-Critic algorithms
;; Naive AC: update neural networks on each timestamp
to go-nac
  ; get current states
  ask turtles [
    set state map runresult inputs
  ]
  let turtle-list sort turtles
  py:set "states" map [ t -> [ state ] of t ] turtle-list
  ;py:run "print(np.array(states).shape)"

  ; using Actor network to choose actions, using e-greedy
  let actions py:runresult "np.argmax(Actor_network.predict(np.array(states)), axis = 1)"
  (foreach turtle-list actions [ [t a] ->
    ask t [
      ifelse (0.05 + random-float 1) < get-epsilon [
        set action random 4
        while [action = 3 and patience > 0] [set action random 4]
      ] [
        set action a
      ]
    ]
  ])

  ; perform chosen actions, i.e. forward cars and get next state
  ask turtles [
    move-forward-rl
    set next-state map runresult inputs
  ]

  py:set "actions" map [ t -> [ action ] of t ] turtle-list
  py:set "rewards" map [ t -> [ reward ] of t ] turtle-list
  py:set "next_states" map [ t -> [ next-state ] of t ] turtle-list

  ; calculate losses of Actor and Critic
  ; - Actor loss is negative of Log probability of action taken multiplied by temporal difference used in Q-learning
  ; - Critic loss is square of the temporal difference
  (py:run
    "with tf.GradientTape() as tape1, tf.GradientTape() as tape2:"
    "   policies = Actor_network(np.array(states))"
    "   values = Critic_network(np.array(states))"
    "   values_next = Critic_network(np.array(next_states))"
    "   tds = rewards + gamma*values_next - values"
    "   dists = tfp.distributions.Categorical(probs=policies, dtype=tf.float32)"
    "   log_probs = dists.log_prob(actions)"
    "   actor_losses = -log_probs*tds"
    "   critic_losses = tds**2"
    "grads1 = tape1.gradient(actor_losses, Actor_network.trainable_variables)"
    "grads2 = tape2.gradient(critic_losses, Critic_network.trainable_variables)"
    "a_optimizer.apply_gradients(zip(grads1, Actor_network.trainable_variables))"
    "c_optimizer.apply_gradients(zip(grads2, Critic_network.trainable_variables))")
end

;; go for A2C algorithms (A2C WITHOUT multiple workers, A2C with multiple workers)
;; A2C: update neural networks after every n timestamp
to go-a2c
  ; get current states
  ask turtles [
    set state map runresult inputs
  ]
  let turtle-list sort turtles
  py:set "states" map [ t -> [ state ] of t ] turtle-list

  ; using Actor network to choose actions, using e-greedy
  let actions py:runresult "np.argmax(Actor_network.predict(np.array(states)), axis = 1)"
  (foreach turtle-list actions [ [t a] ->
    ask t [
      ifelse (0.05 + random-float 1) < get-epsilon [
        set action random 4
        while [action = 3 and patience > 0] [set action random 4]
      ] [
        set action a
      ]
    ]
  ])

  ; perform chosen actions, i.e. forward cars and get next state
  ask turtles [
    move-forward-rl
    set next-state map runresult inputs
  ]

  ; stores current state, action, reward for each turtle
  py:run "step = step + 1"
  ask turtles [
    py:set "id" id
    py:set "state" state
    py:run "mem_states[id].append(state)"
    py:set "action" action
    py:run "mem_actions[id].append(action)"
    py:set "reward" reward
    py:run "mem_rewards[id].append(reward)"

    (py:run
      "if ((step % update_steps) == (id * update_offset)):"
      "   mem_returns[id] = []"
      "   sum_reward = 0"
      "   mem_rewards[id].reverse()"
      "   for r in mem_rewards[id]:"
      "      sum_reward = r + gamma*sum_reward"
      "      mem_returns[id].append(sum_reward)"
      "   mem_returns[id].reverse()"
      "   mem_states[id] = np.array(mem_states[id], dtype=np.float32)"
      "   mem_actions[id] = np.array(mem_actions[id], dtype=np.float32)"
      "   mem_returns[id] = np.array(mem_returns[id], dtype=np.float32)"
      "   mem_returns[id] = tf.reshape(mem_returns[id], (len(mem_returns[id]),))"
      ;; learn function and loss
      "   with tf.GradientTape() as tape1, tf.GradientTape() as tape2:"
      "      policy = Actor_network(mem_states[id], training=True)"
      "      value = Critic_network(mem_states[id], training=True)"
      "      value = tf.reshape(value, (len(value),))"
      "      td = tf.math.subtract(mem_returns[id], value)"
      "      probability = []"
      "      log_probability = []"
      "      for pb, a in zip(policy, mem_actions[id]):"
      "         dist = tfp.distributions.Categorical(probs=pb, dtype=tf.float32)"
      "         log_prob = dist.log_prob(a)"
      "         prob = dist.prob(a)"
      "         probability.append(prob)"
      "         log_probability.append(log_prob)"
      "      p_loss = []"
      "      e_loss = []"
      "      td = td.numpy()"
      "      for pb, t, lpb in zip(probability, td, log_probability):"
      "         t =  tf.constant(t)"
      "         policy_loss = tf.math.multiply(lpb, t)"
      "         entropy_loss = tf.math.negative(tf.math.multiply(pb,lpb))"
      "         p_loss.append(policy_loss)"
      "         e_loss.append(entropy_loss)"
      "      p_loss = tf.stack(p_loss)"
      "      e_loss = tf.stack(e_loss)"
      "      p_loss = tf.reduce_mean(p_loss)"
      "      e_loss = tf.reduce_mean(e_loss)"
      "      actor_loss = -p_loss - 0.0001*e_loss"
      "      critic_loss = 0.5*kls.mean_squared_error(mem_returns[id], value)"
      "   grads1 = tape1.gradient(actor_loss, Actor_network.trainable_variables)"
      "   grads2 = tape2.gradient(critic_loss, Critic_network.trainable_variables)"
      "   a_optimizer.apply_gradients(zip(grads1, Actor_network.trainable_variables))"
      "   c_optimizer.apply_gradients(zip(grads2, Critic_network.trainable_variables))"
      ;; reset memory: states, actions, rewards
      "   mem_states[id] = []"
      "   mem_actions[id] = []"
      "   mem_rewards[id] = []")
  ]

  ;ifelse multiple-workers? [
    ; A2C WITH multiple workers

  ;][
    ; A2C WITHOUT multiple workers
  ;]
end

;;go for REINFORCE
to go-reinforce
  ; get current states
  ask turtles [
    set state map runresult inputs
  ]
  let turtle-list sort turtles
  py:set "states" map [ t -> [ state ] of t ] turtle-list

  let actions py:runresult "np.argmax(R_network.predict(np.array(states)), axis = 1)"
  (foreach turtle-list actions [ [t a] ->
    ask t [
      ifelse (0.05 + random-float 1) < get-epsilon [
        set action random 4
        while [action = 3 and patience > 0] [set action random 4]
      ] [
        set action a
      ]
    ]
  ])

  ; perform chosen actions, i.e. forward cars and get next state
  ask turtles [
    move-forward-rl
    set next-state map runresult inputs
  ]

   ; stores current state, action, reward for each turtle
  ask turtles [
    py:set "id" id
    py:set "state" state
    py:set "action" action
    py:set "reward" reward

    ifelse pxcor != init-xcor [
      py:run "mem_states[id].append(state)"
      py:run "mem_actions[id].append(action)"
      py:run "mem_rewards[id].append(reward)"
    ][
      print word "Update car: "  id
      ; return the initial x coordinate
      (py:run
        "mem_returns[id] = []"
        "sum_reward = 0"
        "mem_returns[id].reverse()"
        "for r in mem_returns[id]:"
        "   sum_reward = r + gamma*sum_reward"
        "   mem_returns[id].append(sum_reward)"
        "mem_returns[id].reverse()"
        "mem_states[id] = np.array(mem_states[id], dtype=np.float32)"
        "mem_actions[id] = np.array(mem_actions[id], dtype=np.float32)"
        "mem_returns[id] = np.array(mem_returns[id], dtype=np.float32)"
        "for a_state, a_return, a_action in zip(mem_states[id], mem_returns[id], mem_actions[id]):"
        "   with tf.GradientTape() as tape:"
        "      prob = R_network(np.array([a_state]), training=True)"
        "      dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)"
        "      log_prob = dist.log_prob(a_action)"
        "      loss = -log_prob*a_return"
        "   grads = tape.gradient(loss, R_network.trainable_variables)"
        "   optimizer.apply_gradients(zip(grads, R_network.trainable_variables))"
        ;; reset
        "mem_states[id] = []"
        "mem_actions[id] = []"
        "mem_rewards[id] = []"
      )
    ]
  ]
end

;; forward car according greedy algorithm
to move-forward-greedy ; turtle procedure
  if speed > 0 [
    ifelse damaged-rate <= random-float 1 [
      set heading 90
      set color get-car-default-color self

      speed-up-car ; we tentatively speed up, but might have to slow down

      let blocking-y-cars other turtles in-cone (blocking-car-observation + speed) 180 with [ get-x-distance <= blocking-car-observation ]
      let blocking-y-car min-one-of blocking-y-cars [ distance myself ]
      if blocking-y-car != nobody [
        ifelse [ speed ] of blocking-y-car > 0 [
          ;while [speed >= ([ speed ] of blocking-y-car)] [slow-down-car]
          set speed [ speed ] of blocking-y-car ; match the speed of the car ahead of you and then slow down so you are driving a bit slower than that car.
          slow-down-car
        ][  ; if blocking-car is a damaged car, with speed = 0
          while [speed > speed-sudden-stop] [slow-down-car]
          ; set speed speed-sudden-stop
          while [ycor = target-lane] [
            choose-new-lane
            if ycor != target-lane [
              move-to-target-lane
            ]
          ]
        ]
      ]

      forward speed

      if patience <= 0  [ choose-new-lane ]  ; want to change lane
      if ycor != target-lane [ move-to-target-lane ]
    ][
      if (count turtles with [speed = 0]) < max-damaged-cars[ ; this car will stop
        set speed 0
        set color damaged-color
      ]
    ]
  ]
end

;; forward car according rl algorithms
to move-forward-rl ; turtle procedure
  if speed > 0 [
    ifelse damaged-rate <= random-float 1  [
      set heading 90

      if action = 0 [ speed-up-car set color green ]   ; 0 = accelerate, green
      if action = 1 [ set color yellow ]               ; 1 = stay same, yellow
      if action = 2 [ slow-down-car set color red]     ; 2 = decelerate, red
      if action = 3 [                                  ; 3 = change lane, blue
        choose-new-lane
        set color blue
        if ycor != target-lane [
          move-to-target-lane
        ]
      ]

      let blocking-y-cars other turtles in-cone (blocking-car-observation + speed) 180 with [ get-y-distance <= blocking-car-observation ]
      let blocking-y-car min-one-of blocking-y-cars [ distance myself ]
      if blocking-y-car != nobody [
        ifelse [ speed ] of blocking-y-car > 0 [
          set speed [ speed ] of blocking-y-car ; match the speed of the car ahead of you and then slow down so you are driving a bit slower than that car.
          slow-down-car
        ][  ; if blocking-car is a damaged car, with speed = 0
          while [speed > speed-sudden-stop] [slow-down-car]
          ;set speed speed-sudden-stop
          while [ycor = target-lane] [
            choose-new-lane
            if ycor != target-lane [
              move-to-target-lane
            ]
          ]
        ]
      ]

      set reward (log (speed + patience / max-patience + 1e-8) 2) ; reward can be positive or negative

      forward speed
    ][
      if (count turtles with [speed = 0]) < max-damaged-cars [ ; this car will stop
        set speed 0
        set action -1
        set color damaged-color
      ]
    ]
  ]
end

;; decrease the value of speed and patience
to slow-down-car ; turtle procedure
  if speed > deceleration [
    set speed speed - deceleration
    if speed < speed-min [ set speed speed-min ]
    set patience patience - 1 ; every time you hit the brakes, you loose a little patience
    if patience < 0 [set patience 0]
  ]
end

;; increase the value of speed and patience
to speed-up-car ; turtle procedure
  set speed speed + acceleration
  if speed > speed-top [ set speed speed-top ]
  ;set patience patience + 1
  ;if patience > max-patience [set patience max-patience]
end

;; convert a state to an integer value
to-report convert-state-int [ aState ]
  ; state-size = max-patience * 101 * world-width * 101
  report (item 0 aState) * 101 * world-width * 101 + (item 1 aState) * world-width * 101 + (item 2 aState) * 101 + (item 3 aState)
end

to create-or-remove-cars
  ; make sure we don't have too many cars for the room we have on the road
  let road-patches patches with [ member? pycor lanes ]
  if number-of-cars > count road-patches [
    set number-of-cars count road-patches
  ]

  let speed-seed 0.3

  create-turtles (number-of-cars - count turtles) [
    set size 0.6
    set color get-car-default-color self
    move-to one-of free road-patches
    set target-lane pycor
    set heading 90
    set speed speed-seed + random-float (speed-max / 2 - speed-seed)
    set speed-top  (2 * speed + random-float (speed-max - 2 * speed)) ;(speed-max / 2) + random-float (speed-max / 2)
    set patience (max-patience / 2) + random (max-patience / 2)
    set action -1
    set reward 0
    set init-xcor pxcor
    set id car-number
    set car-number car-number + 1
  ]

  if count turtles > number-of-cars [
    let n count turtles - number-of-cars
    ask n-of n [ other turtles ] of selected-car [ die ]
  ]

end

to-report free [ road-patches ] ; turtle procedure
  let this-car self
  report road-patches with [
    not any? turtles-here with [ self != this-car ]
  ]
end

to draw-road
  ask patches [
    set pcolor brown - random-float 0.5 ; the road is surrounded by brown ground of varying shades
  ]
  set lanes n-values number-of-lanes [ n -> number-of-lanes - (n * 2) - 1 ]
  ask patches with [ abs pycor <= number-of-lanes ] [
    ; the road itself is varying shades of grey
    set pcolor grey - 2.5 + random-float 0.25
  ]
  draw-road-lines
end

to draw-road-lines
  let y (last lanes) - 1 ; start below the "lowest" lane
  while [ y <= first lanes + 1 ] [
    if not member? y lanes [
      ; draw lines on road patches that are not part of a lane
      ifelse abs y = number-of-lanes
        [ draw-line y yellow 0 ]  ; yellow for the sides of the road
        [ draw-line y white 0.5 ] ; dashed white between lanes
    ]
    set y y + 1 ; move up one patch
  ]
end

;; We use a temporary turtle to draw the line:
;; - with a gap of zero, we get a continuous line;
;; - with a gap greater than zero, we get a dasshed line.
to draw-line [ y line-color gap ]
  create-turtles 1 [
    setxy (min-pxcor - 0.5) y
    hide-turtle
    set color line-color
    set heading 90
    repeat world-width [
      pen-up
      forward gap
      pen-down
      forward (1 - gap)
    ]
    die
  ]
end

;; choose a new lane among those with the minimum distance to your current lane (i.e., your ycor).
to choose-new-lane ; turtle procedure
  let other-lanes remove ycor lanes
  if not empty? other-lanes [
    let min-dist min map [ y -> abs (y - ycor) ] other-lanes
    let closest-lanes filter [ y -> abs (y - ycor) = min-dist ] other-lanes
    set target-lane one-of closest-lanes
    set patience max-patience
  ]
end

to move-to-target-lane ; turtle procedure
  set heading ifelse-value target-lane < ycor [ 180 ] [ 0 ]
  let blocking-x-cars other turtles in-cone (blocking-car-observation + abs (ycor - target-lane)) 180 with [ get-y-distance <= blocking-car-observation ]
  let blocking-x-car min-one-of blocking-x-cars [ distance myself ]
  ifelse blocking-x-car = nobody [
    forward change-lane-fd
    set ycor precision ycor 1 ; to avoid floating point errors
  ][
    ; slow down if the car blocking us is behind, otherwise speed up
    ifelse towards blocking-x-car <= 180 [ slow-down-car ] [ speed-up-car ]
  ]
  set color get-car-default-color self
end

;; allow the user to select a different car by clicking on it with the mouse
to select-car
  if mouse-down? [
    let mx mouse-xcor
    let my mouse-ycor
    if any? turtles-on patch mx my [
      set selected-car one-of turtles-on patch mx my
      ask selected-car [ set color red ]
      display
    ]
  ]
end

to-report get-x-distance
  report distancexy [ xcor ] of myself ycor
end

to-report get-y-distance
  report distancexy xcor [ ycor ] of myself
end

;; calculate the current exploration rate or epsilon
to-report get-epsilon
  ifelse move-strategy != "Greedy" [
    report exp-rate / (1 + exp-decay * ticks)
  ][
    report -1
  ]
end

;; give all cars a blueish color, but still make them distinguishable
to-report get-car-default-color [aCar]
  if aCar = selected-car [ report pink ]
  report white - random-float 5.0  ;one-of [ blue cyan sky ] + 1.5 + random-float 1.0
end

to-report get-distance-to-car [aCar]
  ifelse aCar != nobody [
    report distance aCar
  ][
    report observation-max
  ]
end

to-report get-speed-to-car [aCar]
  ifelse aCar != nobody
  [
    report [speed] of aCar
  ][
    report speed-max
  ]
end

;; report the car ahead that blocks this car in the same lane or in the neighbor lanes
to-report get-blocking-car
  let i 1
  while [ i < observation-max ]
  [
    let blocking-cars other turtles in-cone i 180 with [ get-y-distance <= 1 ]
    let blocking-car min-one-of blocking-cars [ distance myself ]
    ifelse blocking-car != nobody [
      report blocking-car
    ][
      set i (i + 1)
    ]
  ]
  report nobody
end

;; report the car ahead that blocks this car in the same lane
to-report get-car-ahead-same-lane
  let here min-one-of turtles-here with [ xcor > [ xcor ] of myself ] [ distance myself ]
  ifelse here != nobody [
    report here
  ][
    let i 1
    while [ i < observation-max ]
    [
      ifelse (turtles-on patch-ahead i) != nobody [
        report min-one-of turtles-on patch-ahead i [ distance myself ]
      ][
        set i (i + 1)
      ]
    ]
    report nobody
  ]
end

;; report the car behind that blocks this car in the same lane
to-report get-car-behind-same-lane
  let here min-one-of turtles-here with [ xcor < [ xcor ] of myself ] [ distance myself ]
  ifelse here != nobody [
    report here
  ][
    let i 1
    while [ i < observation-max ]
    [
      ifelse (turtles-on patch-ahead (0 - i)) != nobody [
        report min-one-of turtles-on patch-ahead (0 - i) [ distance myself ]
      ][
        set i (i + 1)
      ]
    ]
    report nobody
  ]
end

;to-report get-number-of-lanes
  ; To make the number of lanes easily adjustable, remove this
  ; reporter and create a slider on the interface with the same
  ; name. 8 lanes is the maximum that currently fit in the view.
;  report 3
;end
@#$#@#$#@
GRAPHICS-WINDOW
250
60
1533
597
-1
-1
31.1
1
10
1
1
1
0
1
0
1
-20
20
-8
8
1
1
1
ticks
30.0

BUTTON
20
10
85
45
NIL
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
90
10
155
45
go
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

BUTTON
160
10
225
45
go once
go
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

BUTTON
670
15
780
48
select car
select-car
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

MONITOR
1540
420
1650
465
average speed
(mean [speed] of turtles with [speed > 0]) * 100
2
1
11

SLIDER
5
60
125
93
number-of-cars
number-of-cars
1
number-of-lanes * world-width
60.0
1
1
NIL
HORIZONTAL

PLOT
1160
600
1535
805
Car Speeds
Time
Speed
0.0
300.0
0.0
0.5
true
true
"" ""
PENS
"average" 1.0 0 -2674135 true "" "plot mean [ speed ] of turtles with [speed > 0]"
"max" 1.0 0 -10899396 true "" "plot max [ speed ] of turtles with [speed > 0]"
"min" 1.0 0 -13345367 true "" "plot min [ speed ] of turtles with [speed > 0]"

SLIDER
5
100
125
133
acceleration
acceleration
0.001
0.01
0.009
0.001
1
NIL
HORIZONTAL

SLIDER
130
100
245
133
deceleration
deceleration
0.001
0.1
0.02
0.01
1
NIL
HORIZONTAL

PLOT
1540
240
1880
415
Driver's Patience
Time
Patience
0.0
10.0
0.0
10.0
true
true
"set-plot-y-range 0 max-patience" ""
PENS
"average" 1.0 0 -2674135 true "" "plot mean [ patience ] of turtles"
"max" 1.0 0 -10899396 true "" "plot max [ patience ] of turtles"
"min" 1.0 0 -13345367 true "" "plot min [ patience ] of turtles"

BUTTON
785
15
895
48
follow selected car
follow selected-car
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

BUTTON
900
15
1010
48
watch selected car
watch selected-car
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

BUTTON
1015
15
1125
48
reset perspective
reset-perspective
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

PLOT
1540
60
1880
235
Cars Per Lane
Time
Cars
0.0
0.0
0.0
0.0
true
true
"set-plot-y-range (floor (count turtles * 0.4)) (ceiling (count turtles * 0.6))\nforeach range length lanes [ i ->\n  create-temporary-plot-pen (word \"Lane \" (i + 1))\n  set-plot-pen-color item i base-colors\n]" "foreach range length lanes [ i ->\n  set-current-plot-pen (word \"Lane \" (i + 1))\n  plot count turtles with [ round ycor = item i lanes ]\n]"
PENS

SLIDER
5
140
125
173
max-patience
max-patience
10
100
50.0
1
1
NIL
HORIZONTAL

CHOOSER
5
410
245
455
move-strategy
move-strategy
"Greedy" "Deep Q-Learning" "Deep Q-Network" "Naive Actor-Critic" "Advantage Actor-Critic" "Reinforce"
5

SLIDER
130
60
245
93
number-of-lanes
number-of-lanes
2
15
6.0
1
1
NIL
HORIZONTAL

SLIDER
5
190
125
223
exp-rate
exp-rate
0
1
1.0
0.01
1
NIL
HORIZONTAL

SLIDER
130
190
245
223
exp-decay
exp-decay
0
0.1
0.0
0.001
1
NIL
HORIZONTAL

SLIDER
5
230
125
263
learning-rate
learning-rate
0
0.01
0.0034
0.0001
1
NIL
HORIZONTAL

SLIDER
130
230
245
263
discount-factor
discount-factor
0.80
0.99
0.93
0.01
1
NIL
HORIZONTAL

MONITOR
1540
470
1650
515
exploration rate
get-epsilon
2
1
11

PLOT
250
600
790
805
Selected Actions
Time
# Cars
0.0
800.0
0.0
100.0
true
true
"set-plot-y-range 0 number-of-cars" ""
PENS
"accelerate" 1.0 2 -10899396 true "" "plot count turtles with [ action = 0 ]"
"decelerate" 1.0 2 -2674135 true "" "plot count turtles with [ action = 2 ]"
"stay same" 1.0 2 -1184463 true "" "plot count turtles with [ action = 1 ]"
"change lane" 1.0 2 -13345367 true "" "plot count turtles with [ action = 3 ]"

PLOT
795
600
1155
805
Total Reward
Time
Reward
0.0
300.0
0.0
1.0
true
false
"" ""
PENS
"reward" 1.0 0 -2674135 true "" "plot sum [reward] of turtles"

MONITOR
1655
470
1760
515
total reward
sum [reward] of turtles
2
1
11

MONITOR
1655
420
1760
465
average patience
mean [ patience ] of turtles
2
1
11

MONITOR
1540
520
1715
565
% accelerated action
(count turtles with [ action = 0 ]) / number-of-cars * 100
2
1
11

MONITOR
1720
520
1880
565
% decelerated action
(count turtles with [ action = 2 ]) / number-of-cars * 100
2
1
11

MONITOR
1540
570
1715
615
% stay-same action
(count turtles with [ action = 1 ]) / number-of-cars * 100
2
1
11

MONITOR
1720
570
1880
615
% change-lane action
(count turtles with [ action = 3 ]) / number-of-cars * 100
2
1
11

MONITOR
1765
420
1880
465
damaged cars
count turtles with [speed = 0]
2
1
11

TEXTBOX
10
280
190
311
Parameters for DQN
12
0.0
1

SLIDER
5
300
125
333
batch-size
batch-size
0
1024
128.0
32
1
NIL
HORIZONTAL

SLIDER
130
300
245
333
memory-size
memory-size
0
1000000
17000.0
1000
1
NIL
HORIZONTAL

SWITCH
5
340
125
373
input-exp?
input-exp?
0
1
-1000

SWITCH
130
340
245
373
input-time?
input-time?
0
1
-1000

SLIDER
130
140
245
173
max-damaged-cars
max-damaged-cars
0
5
5.0
1
1
NIL
HORIZONTAL

SLIDER
5
460
245
493
simulation-time
simulation-time
10000
100000
50000.0
1000
1
NIL
HORIZONTAL

@#$#@#$#@
## WHAT IS IT?

This model is a more sophisticated two-lane version of the "Traffic Basic" model.  Much like the simpler model, this model demonstrates how traffic jams can form. In the two-lane version, drivers have a new option; they can react by changing lanes, although this often does little to solve their problem.

As in the Traffic Basic model, traffic may slow down and jam without any centralized cause.

## HOW TO USE IT

Click on the SETUP button to set up the cars. Click on GO to start the cars moving. The GO ONCE button drives the cars for just one tick of the clock.

The NUMBER-OF-CARS slider controls the number of cars on the road. If you change the value of this slider while the model is running, cars will be added or removed "on the fly", so you can see the impact on traffic right away.

The SPEED-UP slider controls the rate at which cars accelerate when there are no cars ahead.

The SLOW-DOWN slider controls the rate at which cars decelerate when there is a car close ahead.

The MAX-PATIENCE slider controls how many times a car can slow down before a driver loses their patience and tries to change lanes.

You may wish to slow down the model with the speed slider to watch the behavior of certain cars more closely.

The SELECT CAR button allows you to highlight a particular car. It turns that car red, so that it is easier to keep track of it. SELECT CAR is easier to use while GO is turned off. If the user does not select a car manually, a car is chosen at random to be the "selected car".

You can either [`watch`](http://ccl.northwestern.edu/netlogo/docs/dictionary.html#watch) or [`follow`](http://ccl.northwestern.edu/netlogo/docs/dictionary.html#follow) the selected car using the WATCH SELECTED CAR and FOLLOW SELECTED CAR buttons. The RESET PERSPECTIVE button brings the view back to its normal state.

The SELECTED CAR SPEED monitor displays the speed of the selected car. The MEAN-SPEED monitor displays the average speed of all the cars.

The YCOR OF CARS plot shows a histogram of how many cars are in each lane, as determined by their y-coordinate. The histogram also displays the amount of cars that are in between lanes while they are trying to change lanes.

The CAR SPEEDS plot displays four quantities over time:

- the maximum speed of any car - CYAN
- the minimum speed of any car - BLUE
- the average speed of all cars - GREEN
- the speed of the selected car - RED

The DRIVER PATIENCE plot shows four quantities for the current patience of drivers: the max, the min, the average and the current patience of the driver of the selected car.

## THINGS TO NOTICE

Traffic jams can start from small "seeds." Cars start with random positions. If some cars are clustered together, they will move slowly, causing cars behind them to slow down, and a traffic jam forms.

Even though all of the cars are moving forward, the traffic jams tend to move backwards. This behavior is common in wave phenomena: the behavior of the group is often very different from the behavior of the individuals that make up the group.

Just as each car has a current speed, each driver has a current patience. Each time the driver has to hit the brakes to avoid hitting the car in front of them, they loose a little patience. When a driver's patience expires, the driver tries to change lane. The driver's patience gets reset to the maximum patience.

When the number of cars in the model is high, drivers lose their patience quickly and start weaving in and out of lanes. This phenomenon is called "snaking" and is common in congested highways. And if the number of cars is high enough, almost every car ends up trying to change lanes and the traffic slows to a crawl, making the situation even worse, with cars getting momentarily stuck between lanes because they are unable to change. Does that look like a real life situation to you?

Watch the MEAN-SPEED monitor, which computes the average speed of the cars. What happens to the speed over time? What is the relation between the speed of the cars and the presence (or absence) of traffic jams?

Look at the two plots. Can you detect discernible patterns in the plots?

The grass patches on each side of the road are all a slightly different shade of green. The road patches, to a lesser extent, are different shades of grey. This is not just about making the model look nice: it also helps create an impression of movement when using the FOLLOW SELECTED CAR button.

## THINGS TO TRY

What could you change to minimize the chances of traffic jams forming, besides just the number of cars? What is the relationship between number of cars, number of lanes, and (in this case) the length of each lane?

Explore changes to the sliders SLOW-DOWN and SPEED-UP. How do these affect the flow of traffic? Can you set them so as to create maximal snaking?

Change the code so that all cars always start on the same lane. Does the proportion of cars on each lane eventually balance out? How long does it take?

Try using the `"default"` turtle shape instead of the car shape, either by changing the code or by typing `ask turtles [ set shape "default" ]` in the command center after clicking SETUP. This will allow you to quickly spot the cars trying to change lanes. What happens to them when there is a lot of traffic?

## EXTENDING THE MODEL

The way this model is written makes it easy to add more lanes. Look for the `number-of-lanes` reporter in the code and play around with it.

Try to create a "Traffic Crossroads" (where two sets of cars might meet at a traffic light), or "Traffic Bottleneck" model (where two lanes might merge to form one lane).

Note that the cars never crash into each other: a car will never enter a patch or pass through a patch containing another car. Remove this feature, and have the turtles that collide die upon collision. What will happen to such a model over time?

## NETLOGO FEATURES

Note the use of `mouse-down?` and `mouse-xcor`/`mouse-ycor` to enable selecting a car for special attention.

Each turtle has a shape, unlike in some other models. NetLogo uses `set shape` to alter the shapes of turtles. You can, using the shapes editor in the Tools menu, create your own turtle shapes or modify existing ones. Then you can modify the code to use your own shapes.

## RELATED MODELS

- "Traffic Basic": a simple model of the movement of cars on a highway.

- "Traffic Basic Utility": a version of "Traffic Basic" including a utility function for the cars.

- "Traffic Basic Adaptive": a version of "Traffic Basic" where cars adapt their acceleration to try and maintain a smooth flow of traffic.

- "Traffic Basic Adaptive Individuals": a version of "Traffic Basic Adaptive" where each car adapts individually, instead of all cars adapting in unison.

- "Traffic Intersection": a model of cars traveling through a single intersection.

- "Traffic Grid": a model of traffic moving in a city grid, with stoplights at the intersections.

- "Traffic Grid Goal": a version of "Traffic Grid" where the cars have goals, namely to drive to and from work.

- "Gridlock HubNet": a version of "Traffic Grid" where students control traffic lights in real-time.

- "Gridlock Alternate HubNet": a version of "Gridlock HubNet" where students can enter NetLogo code to plot custom metrics.

## HOW TO CITE

If you mention this model or the NetLogo software in a publication, we ask that you include the citations below.

For the model itself:

* Wilensky, U. & Payette, N. (1998).  NetLogo Traffic 2 Lanes model.  http://ccl.northwestern.edu/netlogo/models/Traffic2Lanes.  Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.

Please cite the NetLogo software as:

* Wilensky, U. (1999). NetLogo. http://ccl.northwestern.edu/netlogo/. Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.

## COPYRIGHT AND LICENSE

Copyright 1998 Uri Wilensky.

![CC BY-NC-SA 3.0](http://ccl.northwestern.edu/images/creativecommons/byncsa.png)

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.  To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/3.0/ or send a letter to Creative Commons, 559 Nathan Abbott Way, Stanford, California 94305, USA.

Commercial licenses are also available. To inquire about commercial licenses, please contact Uri Wilensky at uri@northwestern.edu.

This model was created as part of the project: CONNECTED MATHEMATICS: MAKING SENSE OF COMPLEX PHENOMENA THROUGH BUILDING OBJECT-BASED PARALLEL MODELS (OBPML).  The project gratefully acknowledges the support of the National Science Foundation (Applications of Advanced Technologies Program) -- grant numbers RED #9552950 and REC #9632612.

This model was converted to NetLogo as part of the projects: PARTICIPATORY SIMULATIONS: NETWORK-BASED DESIGN FOR SYSTEMS LEARNING IN CLASSROOMS and/or INTEGRATED SIMULATION AND MODELING ENVIRONMENT. The project gratefully acknowledges the support of the National Science Foundation (REPP & ROLE programs) -- grant numbers REC #9814682 and REC-0126227. Converted from StarLogoT to NetLogo, 2001.

<!-- 1998 2001 Cite: Wilensky, U. & Payette, N. -->
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.3.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
1
@#$#@#$#@
