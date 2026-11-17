# Przyspieszenie analizy GearBox: co się zmieniło i ile to daje

Analiza zgłoszenia działa teraz mniej więcej **1,8× szybciej na stagingu** (AWS Lambda, 2 vCPU) i **1,5× szybciej na sprzęcie jak produkcja** (4 vCPU, Cloud Run). Na prawdziwych zdjęciach z częściami dochodzi jeszcze zysk na dopasowaniu CAD i fuzji: od 1,8× do 2,8×. **Wyniki analizy się nie zmieniają.** Raporty etapów są identyczne bajt w bajt z kodem z `main`, a jedynym wyjątkiem są ostatnie cyfry ramek YOLO (szczegóły niżej).

Katalog `evaluation_function/gearbox/` (kod synchronizowany z aplikacji desktopowej) jest **nietknięty**. Wszystkie zmiany siedzą w nakładkach w `evaluation_function/`.

## Wyniki w skrócie

Czas całego zgłoszenia (8 zdjęć 12 MPx, pełna ścieżka wywołania):

| Środowisko | Zgłoszenie | Przed | Po | Zysk |
|---|---|---|---|---|
| Lambda, 2 vCPU, 3008 MB, bez `/dev/shm` | kolejne | 49–50 s | **27 s** | −45% (×1,8) |
| Lambda, 2 vCPU | pierwsze po zimnym starcie | 59 s | 42 s | −28% (×1,4) |
| 4 vCPU (jak Cloud Run) | kolejne | 33,4 s | **21–22 s** | −35% (×1,55) |
| 4 vCPU | pierwsze | 42 s | 37 s | −11% |

Do tego dochodzą etapy `cad` + `fuse` na scenie z częściami. Zdjęcia testowe z pierwszej tabeli części nie miały, więc tam te etapy trwały ok. 0 s.

| Wariant | cad + fuse | Zysk |
|---|---|---|
| Przed | 15,9 + 16,0 = 31,9 s | — |
| Po, jeden proces (same przyspieszenia z `accel.py`) | 15,2 + 14,8 = 30,0 s | −6% |
| Po, pula 2 procesów (Lambda, 2 vCPU) | 8,4 + 9,4 = 17,8 s | −44% |
| Po, pula 4 procesów (4 vCPU) | 4,7 + 6,8 = 11,5 s | −64% |

## Gdzie dokładnie zyskaliśmy

Czasy etapów z logów (`steps` w statystykach zgłoszenia), Lambda z 2 vCPU:

| Etap | Przed | Po | Skąd zysk |
|---|---|---|---|
| Start procesu analizy | ~4 s | 0,1 s | proces-matka z załadowanymi bibliotekami (pkt 1) |
| `tag` (AprilTag) | 5,6 s | 5,3 s | bez zmian, ten etap już był wydajny |
| `exif` | 1,3 s | 0,9 s | importy załadowane wcześniej |
| `camera` | 0,7 s | 0,0 s | importy załadowane wcześniej |
| `yolo` | 43,9 s | 19,8 s | wątki (pkt 2), model załadowany i rozgrzany (pkt 1) |
| `cad`, `fuse` | (zależnie od części) | patrz tabela wyżej | pula procesów (pkt 3), `accel.py` (pkt 4) |

Na 4 vCPU etap `yolo` spadł z 19,8 do 16,2 s, a importy w `cad` i `fuse` z 2,5 i 2,1 s do 0,1 s.

### 1. Proces-matka z załadowanym modelem (zygota)
**Pliki:** `worker.py` (`--zygote`), `evaluation.py`, `main.py`

**Wcześniej:** każde zgłoszenie startowało nowy proces Pythona. Ten proces od zera importował torch, ultralytics i open3d, ładował YOLO i robił pierwszą predykcję, która jest wyraźnie wolniejsza od kolejnych: na Lambdzie 9,5 s zamiast ok. 4 s.

**Teraz:**
- Przy starcie serwera (interfejs `rpc`) uruchamia się proces-matka. Importuje biblioteki raz, rozgrzewa model małą predykcją i robi test bezpieczeństwa forka.
- Każde zgłoszenie liczy się w **forku** tego procesu, więc start jest praktycznie natychmiastowy. Fork kończy się po zgłoszeniu, więc pamięć wraca do systemu tak jak wcześniej.
- Między zgłoszeniami proces-matka nic nie robi i ma jeden wątek. Dlatego zamrażanie środowiska na Lambdzie mu nie szkodzi. Stary tryb `EDUVISION_WORKER_REUSE=1` właśnie po takim zamrożeniu potrafił się wywalić z SIGABRT.

**Bezpieczniki:**
- Jeśli fork padnie albo proces-matka nie wstanie, to samo zgłoszenie liczy się w świeżym procesie, jak dotąd. Wymaga to co najmniej 5 s zapasu czasu.
- Po 2 awariach tryb wyłącza się na stałe w danej instancji.
- Przy interfejsie `file` (jedno zgłoszenie na proces) i przy GPU (CUDA nie lubi forka) działa stare zachowanie.

### 2. YOLO: liczba wątków
**Pliki:** `worker.py`, `pipeline.py`

**Przyczyna:** `select_device` w ultralytics ustawia `torch.set_num_threads(min(8, cpu_count - 1))`. Na Lambdzie z 2 vCPU to **1 wątek**, więc połowa procesora stała bezczynnie. Na 4 vCPU to 3 wątki.

**Poprawka:** liczba wątków to teraz `min(8, dostępne CPU)`. Dostępne CPU liczymy z affinity i limitu cgroup, a nadpisać je można przez `EDUVISION_TORCH_THREADS`.

**Efekt:** etap `yolo` jest ok. 2× szybszy na Lambdzie.

### 3. Pula procesów, która działa na Lambdzie
**Pliki:** `forkpool.py`, `pipeline.py`

**Problem:** etapy `cad` i `fuse` używają `ProcessPoolExecutor`, który potrzebuje semaforów w `/dev/shm`. Na Lambdzie `/dev/shm` nie ma, więc te etapy liczyły się w jednym procesie.

**Rozwiązanie:** `ForkPool` to mała pula na `os.fork` i socketpair, bez semaforów. Działa wszędzie na Linuksie.
- Pula startuje na początku zgłoszenia, w ok. 16 ms.
- Ma tyle procesów, ile jest CPU, ale nie więcej, niż mieści się w wolnej pamięci: `EDUVISION_JOB_RESERVE_MB` (1200 MB) na samo zgłoszenie plus ok. 150 MB na proces.
- Jeśli któryś proces puli padnie, zadania liczą się dalej w głównym procesie z tym samym wynikiem.

### 4. Szybsze funkcje dopasowania CAD
**Plik:** `accel.py`

Podmieniamy kilka funkcji z `gearbox/` w czasie działania. Każda zamiennik daje **identyczne liczby bit w bit**, co sprawdza `accel_test.py`, także na pełnym dopasowaniu.
- `render_silhouette` i `silhouette_cost`: histogram etykiet (`cv2.calcHist`) liczony tylko w prostokącie rzutu, a nie na całym obrazie.
- `axis_frame`, `T_camera_part`, `build_roi` i zamiana masek z tensora: mniej kopii i alokacji.

`EDUVISION_ACCEL=0` wyłącza wszystkie podmiany.

### 5. Obrazy diagnostyczne, których nikt nie ogląda
**Plik:** `pipeline.py` (`_FilteredCv2`)

Etapy zapisywały i potem wczytywały obrazy debug, których feedback nie pokazuje. Teraz te odczyty i zapisy są pomijane. Obrazy, które trafiają do feedbacku, powstają jak dotąd. Wszystkie obrazy włącza z powrotem `EDUVISION_DEBUG_IMAGES`.

### 6. Równoległe wysyłanie obrazów do feedbacku
**Pliki:** `attachments.py`, `feedback.py`

Obrazy feedbacku są wysyłane do 4 naraz, a nie po kolei. Wynik jest taki sam jak przy wysyłaniu po kolei: przy błędzie pomijamy obrazy po pierwszym nieudanym i zgłaszamy pierwszy błąd. Zysk jest niewielki i zależy od sieci.

### Dockerfile
Zmieniły się tylko komentarze, które opisują nowy tryb. Wartości zmiennych są takie same jak w `main`.

## Zgodność wyników
- Raporty etapów `cad`, `fuse` i `check` są identyczne bajt w bajt z kodem z `main` dla jednego procesu, puli 2 i puli 4. Sprawdzone na Linuksie bez `/dev/shm`.
- Tekst feedbacku jest identyczny przez całą ścieżkę Lambdy (emulator RIE → shimmy → `main.py`).
- **Jedyna różnica:** YOLO liczone na innej liczbie wątków daje inne ostatnie cyfry ramek, rzędu ±0,0002 px. Maski wychodzą identyczne. Ramki (`bbox_xyxy`) nie są używane w dalszych etapach, a `confidence` liczy się tylko przy rozstrzyganiu remisów w głosowaniu klasy. Taka sama różnica jest już dziś między stagingiem a produkcją, bo mają inną liczbę CPU. Kto chce stałą liczbę wątków, ustawia `EDUVISION_TORCH_THREADS`.

## Pamięć
| Środowisko | Przed (szczyt kontenera) | Po |
|---|---|---|
| 2 vCPU | 1454–1522 MB | 1587 MB |
| 4 vCPU | 1937 MB | 1617 MB |

Po zgłoszeniu w `/tmp/eduvision` zostaje ok. 8 KB.

## Przełączniki (do szybkiego wycofania bez zmiany kodu)
| Zmienna | Co robi |
|---|---|
| `EDUVISION_WORKER_MODE=once` | nowy proces na każde zgłoszenie, jak wcześniej |
| `EDUVISION_WORKERS=0` | bez puli procesów: `cad` i `fuse` w jednym procesie |
| `EDUVISION_ACCEL=0` | oryginalne funkcje dopasowania z `gearbox/` |
| `EDUVISION_TORCH_THREADS=N` | stała liczba wątków YOLO |
| `EDUVISION_DEBUG_IMAGES=1` | wszystkie obrazy diagnostyczne, jak wcześniej |

## Jak mierzyliśmy
- Prawdziwych zdjęć przekładni lokalnie nie było, więc:
  - **pełna ścieżka:** 8 syntetycznych zdjęć 12 MPx z AprilTagiem, podawanych jako URL (base64 nie mieści się w limicie 6 MB Lambdy);
  - **etapy `cad`, `fuse` i `check`:** scena z 8 widokami wyrenderowana z modeli CAD.
- Lambdę emulowaliśmy obrazem bazowym z `aws-lambda-rie`:
  - 2 CPU (`--cpuset-cpus=0,1`) i `os.cpu_count() == 2`;
  - `--memory=3008m`;
  - pusty `/dev/shm` tylko do odczytu;
  - `docker pause`/`unpause` między wywołaniami, jak zamrażanie na Lambdzie.
- Wariant 4 vCPU ma zwykłe `/dev/shm`, jak Cloud Run.
- Wyniki porównywaliśmy z kodem wyeksportowanym z `main` (`98aeb60`).

## Testy
Nowe testy:
- `accel_test.py`: podmienione funkcje i pełne dopasowanie są bit w bit takie jak oryginał;
- `forkpool_test.py`: wyniki, wyjątki, śmierć procesu i powrót do jednego procesu;
- `zygote_test.py`: fork daje ten sam wynik co osobny proces; przekroczenie czasu; awaria forka;
- `pipeline_test.py`: filtr obrazów diagnostycznych i rozmiar puli.

`forkpool_test` i `zygote_test` działają tylko na Linuksie, a na Windows są pomijane. Lokalnie przeszło:
- Linux bez `/dev/shm`: 32 testy, 1 pominięty;
- Windows: 25 testów, 7 pominiętych.

## Na co uważać
- **Testowaliśmy na emulatorze**, nie na prawdziwej Lambdzie ani na Cloud Run. Gdyby fork tam nie zadziałał, zgłoszenia przejdą na stary tryb: będą wolniejsze, ale się nie zepsują. Po wdrożeniu na staging warto zajrzeć do logów. Powinno być `worker_mode: fork` i `executor: fork pool (N)`, a nie powinno być `zygote_failure_*`.
- **Dane były syntetyczne.** Dobrze byłoby puścić `test_full_analysis` z `EDUVISION_TEST_IMAGES` na prawdziwym zestawie zdjęć.
- **Pierwsze zgłoszenie po zimnym starcie Lambdy** dalej czeka ok. 14 s na załadowanie bibliotek, bo shimmy uruchamia `main.py` dopiero przy pierwszym wywołaniu.
- **Prawdziwa Lambda przy 3008 MB daje ok. 1,7 vCPU mocy**, a nie pełne 2 jak w emulacji. Staging może więc być trochę wolniejszy niż te pomiary.
- **Po każdej synchronizacji `gearbox/`** (`sync_gearbox.py`) trzeba puścić `accel_test.py`. Pokaże, czy podmiany z `accel.py` nadal pasują do oryginału.

## Pomysły na dalej
Na 2 vCPU procesor jest prawie cały zajęty przez YOLO (74% czasu), więc bez zmiany wyników zostało już niewiele.

**Bez zmiany wyników:**
- **równoległe pobieranie zdjęć** (`inputs.load_images` pobiera po kolei): 0,3–2 s;
- **AprilTag w trakcie ładowania procesu-matki**, tylko przy zimnym starcie: ok. 5 s;
- **na 4 vCPU YOLO w 2 procesach po 2 wątki** zamiast 1 procesu na 4 wątkach: szacunkowo 3–5 s.

**Bez kodu:** więcej pamięci na Lambdzie, czyli więcej vCPU. Przy 4 vCPU zgłoszenie jest ok. 25–30% szybsze, ale ok. 1,7–1,8× droższe.

**Zmieniają wyniki** (trzeba to ocenić na prawdziwych zdjęciach):
- YOLO na `imgsz` 1024 zamiast 1280: ok. 1,5× szybsze YOLO;
- model w INT8;
- AprilTag zaczynający od `quad_decimate` 2.0.

Sprawdziliśmy i odrzuciliśmy:
- **OpenVINO bez INT8:** brak zysku (3,3 vs 3,1 s na zdjęcie);
- **wspólny bufor zdekodowanych zdjęć:** ok. 1% zysku za ok. 290 MB pamięci.
