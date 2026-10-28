
import os
from pathlib import Path

# Główny folder projektu
PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "models" / "best.pt"
# Katalogi wejścia/wyjścia można nadpisać zmiennymi środowiskowymi
# (aplikacja uruchamia etapy dla osobnych sesji w sessions/<czas>/).
INPUT_DIR = Path(os.environ.get("GEARBOX_INPUT_DIR", PROJECT_DIR / "input"))
OUTPUT_DIR = Path(os.environ.get("GEARBOX_OUTPUT_DIR",
                                 PROJECT_DIR / "output"))
SESSIONS_DIR = PROJECT_DIR / "sessions"

# Liczba procesów do obliczeń równoległych (dopasowanie CAD, łączenie
# zdjęć). 0 = liczba wątków procesora. Wynik nie zależy od tej wartości.
PARALLEL_WORKERS = 0

ACCEPTED_DIR = OUTPUT_DIR / "accepted"
REJECTED_DIR = OUTPUT_DIR / "rejected"
APRILTAG_REPORT_PATH = OUTPUT_DIR / "apriltag_report.json"
TAG_DEBUG_DIR = OUTPUT_DIR / "tag_debug"

# Etap 3 — raport metadanych zdjęć
IMAGE_INFO_JSON_PATH = OUTPUT_DIR / "image_info.json"
IMAGE_INFO_CSV_PATH = OUTPUT_DIR / "image_info.csv"

# Etap 4 — parametry kamer i poza kamery względem AprilTag
CAMERA_POSES_PATH = OUTPUT_DIR / "camera_poses.json"
POSE_DEBUG_DIR = OUTPUT_DIR / "pose_debug"

# Etap 5 — detekcje części
DETECTIONS_PATH = OUTPUT_DIR / "detections.json"
MASKS_DIR = OUTPUT_DIR / "masks"
DETECTION_DEBUG_DIR = OUTPUT_DIR / "detection_debug"

# Etap 6 — modele CAD i poza części (pojedyncze zdjęcia)
CAD_DIR = PROJECT_DIR / "cad"
PART_POSES_PATH = OUTPUT_DIR / "part_poses.json"
PART_POSE_DEBUG_DIR = OUTPUT_DIR / "part_pose_debug"

# Modele CAD części (STL w mm, oś symetrii = Z modelu).
#   shape:       "disc" (koło — inicjalizacja z elipsy) lub
#                "cylinder" (walec — inicjalizacja z prostokąta maski)
#   directional: czy kierunek +Z jest rozróżnialny (gear_big: zębnik po +Z);
#                False = część symetryczna także po odwróceniu końcami.
PART_MODELS = {
    0: {"stl": "short_spacer.stl", "shape": "cylinder", "directional": False},
    1: {"stl": "long_spacer.stl", "shape": "cylinder", "directional": False},
    2: {"stl": "big_gear.stl", "shape": "disc", "directional": True,
        "disc_radius_mm": 12.0, "disc_z_mm": 1.0},
    4: {"stl": "driving_gear.stl", "shape": "disc", "directional": False,
        "disc_radius_mm": 3.0, "disc_z_mm": 3.0},
    5: {"stl": "engine.stl", "shape": "cylinder", "directional": False},
}

# Dopasowanie sylwetki CAD do maski.
PART_FIT_ROI_PX = 256          # dłuższy bok obszaru dopasowania
PART_FIT_EXTRA_WEIGHT = 0.5    # kara za sylwetkę CAD na tle (mniejsza,
                               # bo tło może zasłaniać np. rama/wałek)
PART_FIT_NEUTRAL_WEIGHT = 0.2  # kara za sylwetkę CAD na innych częściach
                               # (mogą zasłaniać, ale nie za darmo — inaczej
                               # część "rośnie", przysuwając się do kamery)

# Reguły montażu: część danej klasy osadzona współosiowo na innej.
# Jeśli środek detekcji leży bliżej niż max_axis_distance_mm od osi
# części-rodzica na tym samym zdjęciu, oś jest przejmowana od rodzica,
# a dopasowywane jest tylko położenie wzdłuż osi. W przeciwnym razie
# (np. części rozrzucone) — dopasowanie swobodne.
# rigid=True: połączenie nierozłączne (jedna całość) — jeśli rodzic jest
# na zdjęciach, część ZAWSZE dostaje jego oś i obraca się z nim, bez
# względu na odległość i jakość dopasowania (te trafiają tylko do uwag).
PART_MOUNTS = {
    4: {"parent_class": 5, "max_axis_distance_mm": 12.0,  # driving_gear
        "rigid": True},                                   # = wałek silnika
}

# Etap 7 — łączenie zdjęć (fizyczne części w 3D)
FUSED_PARTS_PATH = OUTPUT_DIR / "fused_parts.json"
FUSION_DEBUG_DIR = OUTPUT_DIR / "fusion_debug"

# Klasy, które model myli między sobą — obserwacje mogą należeć do
# tej samej fizycznej części; ostateczna klasa = głosowanie
# (najczęstsza, przy remisie — większa suma confidence).
CLASS_CONFUSION_GROUPS = [{0, 1}]   # short_spacer <-> long_spacer

FUSION_MAX_RAY_GAP_MM = 3.5      # max odległość promienia od punktu części
FUSION_MAX_COLOR_DELTA = 30.0    # max różnica barwy (a, b) obserwacji
FUSION_MIN_VIEWS = 2             # mniej = część niepotwierdzona
FUSION_MIN_TRIANGULATION_DEG = 5.0  # min kąt między promieniami
FUSION_MAX_STARTS = 4            # liczba startowych orientacji
FUSION_MIN_IOU_OK = 0.7          # średnie IoU po zdjęciach
# Części leżą na płaszczyźnie znacznika (podstawa lub stół): najniższy
# punkt bryły nie może być niżej niż ta wartość (mm, Z świata).
FUSION_MIN_Z_MM = -2.0
FUSION_BELOW_PLANE_WEIGHT = 0.05  # kara za każdy mm poniżej
# Reguła montażu (PART_MOUNTS) jest przyjmowana, jeśli dopasowanie
# z wymuszoną współosiowością jest najwyżej o tyle gorsze.
FUSION_MOUNT_MAX_COST_RATIO = 1.3
# Obserwacje z małych grup (< FUSION_STRONG_VIEWS zdjęć) są dołączane do
# pewnej części, jeśli jej rzut CAD pokrywa maskę z IoU >= progu.
FUSION_STRONG_VIEWS = 3
# Grupy widoczne na mniejszej części zdjęć niż ten ułamek też najpierw
# próbują dołączyć się do pewnych części (np. obserwacje spacera, które
# przez zasłonięcie nie przeszły progu promieni, zamiast tworzyć
# fałszywą część). Gdy się nie da — nadal stają się osobną częścią.
FUSION_STRONG_VIEWS_FRACTION = 0.5
FUSION_ABSORB_MIN_IOU = 0.5
# Obserwacja, której maski model 3D części prawie nie pokrywa
# (IoU poniżej progu), jest od części odłączana i może trafić do innej.
FUSION_EVICT_MAX_IOU = 0.2

# Grupy współosiowe (części na jednym wałku / osi silnika): osie prawie
# równoległe i blisko jednej prostej -> wspólna oś dopasowana do
# wszystkich zdjęć; każda część przesuwa się tylko wzdłuż tej osi.
SHAFT_MAX_ANGLE_DEG = 15.0
SHAFT_MAX_OFFSET_MM = 3.0
SHAFT_MAX_COST_RATIO = 1.3   # przyjmij, jeśli koszt wzrósł najwyżej tyle
# Część, której środek leży tak blisko osi istniejącej grupy, dołącza do
# niej niezależnie od własnej osi (np. krótki spacer — jego orientacja
# z sylwetki jest słabo określona).
SHAFT_ATTACH_MAX_OFFSET_MM = 2.5

# Etap 8 — kontrola montażu
ASSEMBLIES_DIR = PROJECT_DIR / "assemblies"
ASSEMBLY_CHECK_PATH = OUTPUT_DIR / "assembly_check.json"
ASSEMBLY_3D_PATH = OUTPUT_DIR / "assembly_3d.ply"

# Wałki rysowane w podglądzie 3D (tylko wizualnie — nie są częściami
# i nie wpływają na kontrolę). Średnica = otwór kół w cad/*.py (bore 3 mm).
MOTOR_SHAFT_DIAMETER_MM = 3.0    # wałek silnika: od czoła silnika do końca
                                 # osadzonego na nim driving_gear
SHAFT_VISUAL_DIAMETER_MM = 3.0   # wałki kół (wspólne osie S1, S2, ...)
SHAFT_VISUAL_MARGIN_MM = 10.0     # wysunięcie poza skrajne części na wałku

# Uzębienia części (z cad/*.py): sekcje wzdłuż osi Z modelu CAD.
# Średnica podziałowa = moduł * liczba zębów; rozstaw osi pary
# zazębionej = (d1 + d2) / 2.
GEAR_SECTIONS = {
    2: [  # gear_big: koło 24 mm (z=0..2) + zębnik 6 mm (z=2..6)
        {"name": "big", "module": 0.3, "teeth": 78, "z_mm": (0.0, 2.0)},
        {"name": "pinion", "module": 0.3, "teeth": 18, "z_mm": (2.0, 6.0)},
    ],
    4: [  # driving_gear: 6 mm, z=0..6
        {"name": "main", "module": 0.3, "teeth": 18, "z_mm": (0.0, 6.0)},
    ],
}
# Część napędzająca i części sztywno z nią połączone (PART_MOUNTS).
DRIVE_SOURCE_CLASS = 5   # engine

# Tolerancje kontroli (pomiar 3D ma ~0.5 mm niepewności).
CHECK_PARALLEL_TOL_DEG = 6.0
CHECK_COAXIAL_TOL_MM = 1.5
# |rozstaw zmierzony - wymagany|. Dokładność pomiaru położenia kół to
# ok. ±1 mm (zmierzone odchyłki prawdziwych zazębień: +0.15…+0.81 mm),
# a koła niezazębione różnią się rozstawem o >= 9 mm — 1.5 mm rozdziela
# je bezpiecznie.
CHECK_MESH_CENTER_TOL_MM = 1.5
CHECK_MESH_MIN_OVERLAP_MM = 0.5   # wspólny zakres zębów wzdłuż osi
# "adjacent": części na jednej osi, odstęp między bryłami wzdłuż osi
# najwyżej tyle (mm); nakładanie się brył ponad CHECK_MAX_PENETRATION_MM
# to kolizja.
CHECK_ADJACENT_MAX_GAP_MM = 1.0
CHECK_MAX_PENETRATION_MM = 0.5
# Klasa uznawana za niepewną, gdy w głosowaniu klasy (etap 7) zwycięzca
# ma przewagę najwyżej tylu głosów.
CHECK_UNCERTAIN_VOTE_MARGIN = 1
PART_FIT_MAX_EVALS = 400
PART_FIT_MIN_IOU_OK = 0.75
PART_FIT_MIN_IOU = 0.5
PART_FIT_AMBIGUITY_RATIO = 1.15   # koszt alternatywy / koszt najlepszej

# Klasy modelu
CLASS_NAMES = {
    0: "short_spacer",
    1: "long_spacer",
    2: "gear_big",
    3: "gear_sk",
    4: "driving_gear",
    5: "engine",
    6: "gear_small",
}

# gear_sk nie występuje w docelowym montażu
IGNORED_CLASS_IDS = {3}
ACTIVE_CLASS_IDS = [
    class_id
    for class_id in CLASS_NAMES
    if class_id not in IGNORED_CLASS_IDS
]

# Klasy, które nie są osobnymi częściami, tylko cechą innej części:
# {klasa cechy: klasa części-rodzica}. gear_small to zębnik koła
# złożonego gear_big (na środku bocznej ściany) — wskazuje, w którą
# stronę koło jest obrócone. Liczba części nie jest z góry zakładana.
FEATURE_CLASS_PARENTS = {6: 2}

# Klasy o kształcie koła (dopasowanie elipsy do maski).
CIRCULAR_CLASS_IDS = {2, 4, 6}

# Przypisanie cechy do rodzica: odległość środków względem promienia
# rodzica oraz minimalny udział styku masek.
FEATURE_MAX_CENTER_DIST = 0.8
FEATURE_MIN_CONTACT = 0.05
# Waga różnicy koloru (cecha i rodzic to zwykle jeden odlew).
# 0 = ignoruj kolor. Koszt = odległość_norm + waga * ΔE_ab / 50.
FEATURE_COLOR_WEIGHT = 1.0
# Jeśli drugi kandydat ma koszt mniejszy niż najlepszy + margines,
# przypisanie oznaczane jest jako niejednoznaczne.
FEATURE_AMBIGUITY_MARGIN = 0.15

# Na znaczniku AprilTag nie ma części — detekcje leżące na nim to
# fałszywe trafienia (białe pola wzoru). Strefa = czarny kwadrat
# powiększony o SCALE względem środka (1.25 obejmuje biały margines).
TAG_EXCLUSION_SCALE = 1.25
TAG_EXCLUSION_MIN_OVERLAP = 0.5   # udział pola maski w strefie

CONFIDENCE_THRESHOLD = 0.70
IMAGE_SIZE = 1280

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# AprilTag — parametry znacznika
APRILTAG_FAMILY = "tag36h11"
APRILTAG_ID = 0

# Długość boku CZARNEGO kwadratu znacznika w metrach (to jego narożniki
# zwraca detektor). Plik tag36h11-0-30mm-24mm.svg: całość z białym
# marginesem 30 mm, czarny kwadrat 24 mm (viewBox 10..90 ze 100).
# Potwierdzone pomiarem wydruku: 24 x 24 mm.
APRILTAG_SIZE_M = 0.024  # 24 mm

# Kolejne próby detekcji AprilTag (quad_decimate). Na zdjęciach
# 12 MPx lekkie rozmycie psuje wykrywanie czworokątów w pełnej
# rozdzielczości; decymacja 2x to naprawia. Krawędzie i narożniki są
# i tak dopracowywane na obrazie w pełnej rozdzielczości (refine_edges).
APRILTAG_QUAD_DECIMATE_STEPS = (1.0, 2.0, 3.0)

# --- Jakość zdjęć (image_quality.py) -------------------------------------
# Mierzona na znaczniku AprilTag. Dwa poziomy: WARNING (zdjęcie użyte,
# z uwagą) i REJECTED (zdjęcie wykluczone z analizy). Progi wstępne —
# ustalone z zapasem względem zdjęć, na których detekcja działa dobrze
# (do doprecyzowania serią testową: odległość / światło / poruszenie).
# Skala: px na 1 mm mechanizmu w obrazie widzianym przez model YOLO.
QUALITY_MIN_SCALE_WARN = 3.0
QUALITY_MIN_SCALE_REJECT = 2.0
#   Zmierzone na dobrych zdjęciach (2 zestawy, 23 zdjęcia):
#   skala 3.6–9.6 px/mm, rozmycie 0.26–1.35 mm, biel 129–188,
#   kontrast 66–103.
# Rozmycie krawędzi znacznika (10–90%) w mm na obiekcie. Znacznik leży
# na podstawce, a telefon ostrzy na mechanizm — bywa lekko nieostry,
# dlatego progi są łagodne.
QUALITY_MAX_BLUR_MM_WARN = 1.6
QUALITY_MAX_BLUR_MM_REJECT = 2.5
# Jasność białego marginesu znacznika (0–255) i kontrast biel - czerń.
QUALITY_MIN_WHITE_WARN = 100
QUALITY_MIN_WHITE_REJECT = 60
QUALITY_MIN_CONTRAST_WARN = 50
QUALITY_MIN_CONTRAST_REJECT = 30
# Udział prześwietlonej bieli (>= 250) na marginesie znacznika.
QUALITY_MAX_WHITE_CLIPPED = 0.5

# --- Pokrycie widokami (estimate_camera_poses.py) — tylko ostrzeżenia ---
# Liczone względem punktu, na który patrzą aparaty (środek mechanizmu).
# Dobre zestawy (12 i 10 zdjęć): ok. 160° wokół mechanizmu, elewacja
# 48–86°, maks. kąt między widokami 45–57°.
COVERAGE_MIN_IMAGES = 5
COVERAGE_MIN_AZIMUTH_SPAN_DEG = 90.0
COVERAGE_MIN_VIEW_ANGLE_DEG = 25.0
COVERAGE_MIN_ELEVATION_RANGE_DEG = 10.0

# Szerokość matrycy (mm) dla znanych aparatów — tylko do WSTĘPNEGO
# oszacowania ogniskowej w pikselach z EXIF FocalLength. Dane z
# publicznych specyfikacji, niezweryfikowane; kalibracja ma pierwszeństwo.
# Klucz: (Make, Model) dokładnie jak w EXIF.
SENSOR_WIDTH_MM = {
    # Samsung ISOCELL GM1, 8000 x 0.8 µm = 6.4 mm (dłuższy bok)
    ("Xiaomi", "Redmi Note 8T"): 6.4,
}

# --- Etap 4: model kamery -------------------------------------------------
# Założenia modelu (wspólne dla wszystkich telefonów):
#   fx = fy, punkt główny w środku obrazu, brak dystorsji.
# Główne aparaty telefonów zwykle zapisują JPEG już z korekcją dystorsji;
# założenie zostanie zweryfikowane w kolejnych etapach.

# Względna niepewność (1 sigma) ogniskowej z różnych źródeł.
FOCAL_REL_STD_EXIF = 0.05
# Gdy nie ma EXIF: typowy telefon ma f ≈ 0.8 * dłuższy bok obrazu
# (kąt widzenia ~65°), ale rozrzut jest duży.
DEFAULT_FOCAL_FACTOR = 0.8
FOCAL_REL_STD_DEFAULT = 0.25

# Szum położenia narożników AprilTag (1 sigma, px) przyjmowany do
# szacowania niepewności. Używana jest większa z tej wartości
# i zmierzonego błędu reprojekcji.
APRILTAG_CORNER_SIGMA_PX = 0.5

# Kryteria użycia pojedynczego zdjęcia do oszacowania ogniskowej
# z AprilTaga (pomiar kontrolny dla EXIF).
TAG_FOCAL_MAX_VIEW_RMS_PX = 1.0
TAG_FOCAL_MAX_REL_STD = 0.15
# Próg niezgodności EXIF vs AprilTag (w sigmach).
FOCAL_CONSISTENCY_SIGMA = 3.0

# Kryteria jakości pozy kamery.
POSE_MAX_REPROJ_RMS_PX = 1.5
POSE_MIN_AMBIGUITY_RATIO = 3.0     # błąd 2. rozwiązania / błąd 1.
POSE_MAX_FOCAL_REL_STD = 0.15      # powyżej -> UNKNOWN (skala niewiarygodna)