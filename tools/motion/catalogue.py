"""What the Solar Patrol motion library holds: base poses, loops and the moves that join them.

Every move starts on a pinned pose, either a base pose (stand, kneel, crouch, prone) or the first frame of a loop, and
every move that leads somewhere ends pinned on the pose it leads to, so any two moves the director chains meet exactly.
A move with no end target is terminal: the director holds its last frame.

Prompts follow Kimodo's training captions: "A person ...", present tense, one or two actions, the body only (the model
knows no objects and no scene), about 15 to 20 words.
"""

FPS = 30

# Base poses: generated once, the stillest frame of the best take becomes the pinned pose.
HUBS = {
    "stand": "A person stands upright and still with the arms relaxed at the sides, looking straight ahead.",
    "kneel": "A person kneels on the right knee with the left foot forward, both hands resting close to the ground.",
    "crouch": "A person squats down low on both feet with the knees bent and the hands near the ground, staying still.",
    "prone": "A person lies flat face down on the ground with the arms bent beside the head, completely still.",
}

# Loops: stage, prompt, start (a base pose or None), forward speed in m/s for a straight root path (None: in place),
# how many distinct takes to keep. Loops with a start base pose keep their lead-in as the way into the loop.
LOOPS = {
    "walk": ("arrival", "A person walks forward at a steady normal pace with the arms swinging naturally.", None, 1.35, 2),
    "walk_wary": ("arrival", "A person walks forward slowly and cautiously, turning the head to look left and right.", None, 0.9, 2),
    "sneak": ("arrival", "A person sneaks forward in a low crouch, taking slow careful steps and staying quiet.", None, 0.6, 2),
    "run": ("reaction", "A person sprints forward as fast as possible with the arms pumping hard.", None, 4.5, 2),
    "crawl": ("reaction", "A person crawls forward on the hands and knees, keeping the head and back low.", None, 0.35, 2),
    "carry": ("theft", "A person walks forward carrying a heavy load on the right shoulder, holding it steady with one hand.", None, 1.2, 2),
    "cut_fence": ("entry", "A person stands facing forward and squeezes a long tool with both hands at chest height, cutting again and again.", "stand", None, 2),
    "cut_cable": ("theft", "A person kneeling on one knee hammers repeatedly at something on the ground in front of them with both hands.", "kneel", None, 2),
    "pull_cable": ("theft", "A person bent forward pulls something up from the ground hand over hand, stepping slowly backward.", "stand", None, 2),
    "look_around": ("arrival", "A person stands in place and looks around nervously, turning the head and upper body left and right.", "stand", None, 2),
    "crouch_watch": ("reaction", "A person squatting low stays frozen and slowly turns the head to look up and around.", "crouch", None, 1),
    "prone_still": ("reaction", "A person lying face down on the ground stays completely still with the head down.", "prone", None, 1),
}

# Joins and reactions: stage, prompt, start, end, frames. start is a base pose or ("loop", name); end is a base pose,
# ("loop", name) or None for a terminal move. A move that starts on a loop is made once for every kept take of it.
MOVES = {
    # into and out of the base poses
    "stand_to_kneel": ("theft", "A person standing kneels down on the right knee and reaches toward the ground with both hands.", "stand", "kneel", 90),
    "kneel_to_stand": ("theft", "A person kneeling on one knee stands up and looks around carefully.", "kneel", "stand", 90),
    "stand_to_crouch": ("reaction", "A person standing quickly squats down low and stays still.", "stand", "crouch", 75),
    "crouch_to_stand": ("reaction", "A person squatting low slowly stands up and looks around.", "crouch", "stand", 90),
    "stand_to_walk": ("arrival", "A person standing still starts walking forward at a normal pace.", "stand", ("loop", "walk"), 75),
    "stand_to_walk_wary": ("arrival", "A person standing still starts walking forward slowly and cautiously, looking left and right.", "stand", ("loop", "walk_wary"), 75),
    "crouch_to_sneak": ("arrival", "A person crouching low starts walking forward slowly in a fully crouched position, taking careful steps.", "crouch", ("loop", "sneak"), 75),
    "prone_to_crawl": ("reaction", "A person lying on the stomach gets up onto the hands and knees and crawls forward slowly.", "prone", ("loop", "crawl"), 90),
    "prone_to_run": ("reaction", "A person lying face down jumps up to the feet and starts running away fast.", "prone", ("loop", "run"), 90),
    # the walk and its reactions
    "walk_to_stand": ("arrival", "A person walking forward slows down and comes to a stop, standing still.", ("loop", "walk"), "stand", 75),
    "walk_to_kneel": ("theft", "A person walking forward stops and kneels down on the right knee, reaching toward the ground.", ("loop", "walk"), "kneel", 105),
    "walk_freeze": ("reaction", "A person walking forward stops dead, looks up at the sky and stands completely still.", ("loop", "walk"), None, 105),
    "walk_to_prone": ("reaction", "A person walking forward suddenly drops to the ground and lies flat on the stomach.", ("loop", "walk"), "prone", 90),
    "walk_to_run": ("reaction", "A person walking forward suddenly breaks into a fast sprint.", ("loop", "walk"), ("loop", "run"), 60),
    "walk_wary_to_stand": ("arrival", "A person walking cautiously slows down and stops, standing still and looking around.", ("loop", "walk_wary"), "stand", 75),
    "walk_wary_freeze": ("reaction", "A person walking cautiously stops dead, looks up at the sky and stands completely still.", ("loop", "walk_wary"), None, 105),
    "walk_wary_to_run": ("reaction", "A person walking cautiously suddenly breaks into a fast sprint.", ("loop", "walk_wary"), ("loop", "run"), 60),
    "sneak_to_crouch": ("reaction", "A person sneaking forward in a crouch stops and squats low, staying still.", ("loop", "sneak"), "crouch", 75),
    "sneak_to_prone": ("reaction", "A person sneaking forward in a crouch drops flat onto the stomach and lies still.", ("loop", "sneak"), "prone", 90),
    "sneak_to_run": ("reaction", "A person sneaking forward in a crouch jumps up and sprints away fast.", ("loop", "sneak"), ("loop", "run"), 75),
    "carry_freeze": ("reaction", "A person walking with a load on the shoulder stops dead, looks up and stands completely still.", ("loop", "carry"), None, 105),
    "carry_to_run": ("reaction", "A person walking with a load on the shoulder suddenly breaks into a fast run.", ("loop", "carry"), ("loop", "run"), 60),
    "crawl_to_prone": ("reaction", "A person crawling on the hands and knees lowers down flat onto the stomach and lies still.", ("loop", "crawl"), "prone", 75),
    # the work and its reactions
    "cut_fence_to_stand": ("entry", "A person lowers both arms to the sides and stands upright and still, relaxed.", ("loop", "cut_fence"), "stand", 75),
    "cut_fence_to_walk": ("entry", "A person working at chest height ducks the head and steps forward through a narrow gap, then walks on.", ("loop", "cut_fence"), ("loop", "walk"), 105),
    "cut_fence_freeze": ("reaction", "A person working with both hands at chest height stops, looks up at the sky and stays completely still.", ("loop", "cut_fence"), None, 105),
    "cut_fence_to_run": ("reaction", "A person working with both hands at chest height turns around and sprints away fast.", ("loop", "cut_fence"), ("loop", "run"), 75),
    "cut_cable_to_stand": ("theft", "A person kneeling and working near the ground stops and stands up.", ("loop", "cut_cable"), "stand", 90),
    "cut_cable_freeze": ("reaction", "A person kneeling and working near the ground stops, looks up at the sky and stays completely still.", ("loop", "cut_cable"), None, 105),
    "cut_cable_to_prone": ("reaction", "A person kneeling on one knee drops flat onto the stomach and lies still, head down.", ("loop", "cut_cable"), "prone", 90),
    "cut_cable_to_run": ("reaction", "A person kneeling on one knee springs up and sprints away as fast as possible.", ("loop", "cut_cable"), ("loop", "run"), 75),
    "pull_cable_to_stand": ("theft", "A person bent forward pulling something stops, straightens up and stands still.", ("loop", "pull_cable"), "stand", 75),
    "pull_cable_to_carry": ("theft", "A person bent forward lifts a heavy load onto the right shoulder and starts walking forward.", ("loop", "pull_cable"), ("loop", "carry"), 105),
    "pull_cable_freeze": ("reaction", "A person bent forward pulling something stops, looks up at the sky and stays completely still.", ("loop", "pull_cable"), None, 105),
    "pull_cable_to_run": ("reaction", "A person bent forward pulling something drops it, turns and sprints away fast.", ("loop", "pull_cable"), ("loop", "run"), 75),
    "look_around_freeze": ("reaction", "A person standing suddenly stops moving, raises the head to look upward and stays completely still.", ("loop", "look_around"), None, 90),
    "look_around_to_run": ("reaction", "A person looking around nervously suddenly turns and sprints away fast.", ("loop", "look_around"), ("loop", "run"), 75),
    "crouch_watch_to_run": ("reaction", "A person squatting low springs up and sprints away as fast as possible.", ("loop", "crouch_watch"), ("loop", "run"), 75),
    "crouch_watch_to_prone": ("reaction", "A person squatting low drops flat onto the stomach and lies still.", ("loop", "crouch_watch"), "prone", 75),
    # a way out that starts with a step back, so a thief can work right at a panel's edge and still leave it
    "cut_cable_back_away": ("reaction", "A person kneeling on one knee stands up and walks backward a few steps.", ("loop", "cut_cable"), "stand", 90),
    "stand_to_run": ("reaction", "A person standing still suddenly turns around and sprints away as fast as possible.", "stand", ("loop", "run"), 75),
    # a freeze that is not the end: back to work, or off at a run
    "cut_cable_freeze_resume": ("reaction", "A person kneeling and working near the ground freezes and looks up at the sky for a moment, then goes back to working near the ground.", ("loop", "cut_cable"), ("loop", "cut_cable"), 150),
    "cut_cable_freeze_run": ("reaction", "A person kneeling and working near the ground freezes and looks up at the sky, then springs up and sprints away fast.", ("loop", "cut_cable"), ("loop", "run"), 150),
    "cut_fence_freeze_resume": ("reaction", "A person working with both hands at chest height freezes and looks up at the sky for a moment, then goes back to working with both hands.", ("loop", "cut_fence"), ("loop", "cut_fence"), 150),
    "cut_fence_freeze_run": ("reaction", "A person working with both hands at chest height freezes and looks up at the sky, then turns around and sprints away fast.", ("loop", "cut_fence"), ("loop", "run"), 150),
    "pull_cable_freeze_resume": ("reaction", "A person bent forward pulling something freezes and looks up at the sky for a moment, then goes back to pulling hand over hand.", ("loop", "pull_cable"), ("loop", "pull_cable"), 150),
    "pull_cable_freeze_run": ("reaction", "A person bent forward pulling something freezes and looks up at the sky, then drops it, turns and sprints away fast.", ("loop", "pull_cable"), ("loop", "run"), 150),
    "walk_freeze_resume": ("reaction", "A person walking forward stops and looks up at the sky for a moment, then walks on.", ("loop", "walk"), ("loop", "walk"), 150),
    "walk_freeze_run": ("reaction", "A person walking forward stops and looks up at the sky, then suddenly sprints away fast.", ("loop", "walk"), ("loop", "run"), 150),
}

# Candidates generated per kept result: loops draw this many takes and keep the most distinct passing ones; every
# other move draws this many per start and keeps the best passing one.
LOOP_CANDIDATES = 6
HUB_CANDIDATES = 4
MOVE_CANDIDATES = 6
# Moves that failed their checks at six takes draw twelve.
MOVE_CANDIDATES_BY_NAME = {"cut_fence_freeze": 12, "cut_fence_to_stand": 12, "sneak_to_prone": 12, "walk_to_prone": 12,
                           "cut_cable_back_away": 12, "stand_to_run": 12, "cut_cable_freeze_resume": 12,
                           "cut_cable_freeze_run": 12, "cut_fence_freeze_resume": 12, "cut_fence_freeze_run": 12,
                           "pull_cable_freeze_resume": 12, "pull_cable_freeze_run": 12, "walk_freeze_resume": 12,
                           "walk_freeze_run": 12}
LOOP_FRAMES = 300
LOOP_FRAMES_BY_NAME = {"run": 180}  # a sprint held longer than about 6 s freezes into a glide
STEPPING = {"walk", "walk_wary", "sneak", "run", "carry"}  # upright gaits whose feet must lift; a crawl drags its toes
HUB_FRAMES = 120
