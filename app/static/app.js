const state = {
  data: null,
  filterKind: "all",
  filterCategory: "all",
  filterColour: "all",
  lostPhoto: "",
  foundPhoto: "",
};

function shortName(place) {
  return place.short || place.name;
}

function when(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso || "—";
  return date.toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" });
}

function pct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function localNow() {
  const date = new Date();
  const pad = (value) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

async function boot() {
  const response = await fetch("/api/bootstrap");
  if (!response.ok) throw new Error("Could not load the matcher.");
  state.data = await response.json();
  fillHeader(state.data.counts);
  fillShared();
  buildForm("lost");
  buildForm("found");
  renderRecords();
  renderMap("lost-map", state.data.places[0].name, []);
  renderMap("found-map", state.data.places[0].name, []);
}

function fillHeader(counts) {
  document.querySelector("#stat-lost").textContent = String(counts.lost);
  document.querySelector("#stat-found").textContent = String(counts.found);
  document.querySelector("#stat-total").textContent = String(counts.total);
  document.querySelector("#statbar").hidden = false;
  const names = state.data.group.members.map((member) => member.name).join(" · ");
  document.querySelector("#members").textContent = names;
  const weights = state.data.fusion_weights;
  document.querySelector("#formula").innerHTML =
    `Matcher: <strong>${weights.visual.toFixed(2)}</strong> visual + ` +
    `<strong>${weights.location.toFixed(2)}</strong> place + ` +
    `<strong>${weights.time.toFixed(2)}</strong> time`;
}

function fillShared() {
  const filters = document.querySelector("#filters");
  filters.replaceChildren();
  addFilter(filters, "kind", "all", "All records");
  addFilter(filters, "kind", "lost", "Lost");
  addFilter(filters, "kind", "found", "Found");
  addFilter(filters, "category", "all", "All items");
  state.data.categories.forEach((item) => addFilter(filters, "category", item.id, item.label));
  addFilter(filters, "colour", "all", "Any colour");
  state.data.colours.forEach((name) => addFilter(filters, "colour", name, name));
  filters.querySelector("[data-group='kind'][data-value='all']").classList.add("on");
  filters.querySelector("[data-group='category'][data-value='all']").classList.add("on");
  filters.querySelector("[data-group='colour'][data-value='all']").classList.add("on");
}

function addFilter(container, group, value, label) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.dataset.group = group;
  button.dataset.value = value;
  button.addEventListener("click", () => {
    container.querySelectorAll(`[data-group="${group}"]`).forEach((item) => item.classList.remove("on"));
    button.classList.add("on");
    if (group === "kind") state.filterKind = value;
    if (group === "category") state.filterCategory = value;
    if (group === "colour") state.filterColour = value;
    renderRecords();
  });
  container.append(button);
}

function buildForm(kind) {
  const root = document.querySelector(`#${kind}-fields`);
  root.replaceChildren();
  root.append(
    field("Category", select(`${kind}-category`, state.data.categories.map((item) => [item.id, item.label]))),
    colourField(kind),
    field(
      kind === "found" ? "Found at" : "Last seen at",
      select(
        `${kind}-location`,
        state.data.places.map((place) => [place.name, place.name])
      ),
      true
    ),
    field(kind === "found" ? "Found time" : "Lost time", timeInput(`${kind}-time`), true),
    photoField(kind),
    field("Optional note", noteInput(`${kind}-note`), true)
  );
  syncSwatch(kind);
  document.querySelector(`#${kind}-location`).addEventListener("change", (event) => {
    renderMap(`${kind}-map`, event.target.value, []);
  });
}

function field(label, control, wide = false) {
  const wrap = document.createElement("label");
  if (wide) wrap.classList.add("wide");
  wrap.append(label, control);
  return wrap;
}

function select(id, options) {
  const node = document.createElement("select");
  node.id = id;
  node.required = true;
  options.forEach(([value, label]) => node.add(new Option(label, value)));
  return node;
}

function colourField(kind) {
  const wrap = document.createElement("label");
  wrap.append("Colour");
  const row = document.createElement("span");
  row.className = "colour-row";
  const swatch = document.createElement("i");
  swatch.className = "swatch";
  swatch.id = `${kind}-swatch`;
  const node = select(`${kind}-colour`, state.data.colours.map((name) => [name, name]));
  node.addEventListener("change", () => syncSwatch(kind));
  row.append(swatch, node);
  wrap.append(row);
  return wrap;
}

function timeInput(id) {
  const node = document.createElement("input");
  node.id = id;
  node.type = "datetime-local";
  node.required = true;
  node.value = localNow();
  return node;
}

function noteInput(id) {
  const node = document.createElement("input");
  node.id = id;
  node.type = "text";
  node.maxLength = 400;
  node.placeholder = "Where it was left, a name tag, a case colour…";
  return node;
}

function photoField(kind) {
  const wrap = document.createElement("label");
  wrap.className = "wide file";
  wrap.append(kind === "found" ? "Photo of the found item" : "Optional photo of a similar item");
  const input = document.createElement("input");
  input.id = `${kind}-photo`;
  input.type = "file";
  input.accept = "image/*";
  if (kind === "found") input.required = true;
  const name = document.createElement("span");
  name.id = `${kind}-photo-name`;
  name.textContent = "No photo selected";
  input.addEventListener("change", async (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    const data = await shrink(file);
    if (kind === "lost") state.lostPhoto = data;
    else state.foundPhoto = data;
    name.textContent = file.name;
  });
  wrap.append(input, name);
  return wrap;
}

function syncSwatch(kind) {
  const colour = document.querySelector(`#${kind}-colour`).value;
  document.querySelector(`#${kind}-swatch`).className = `swatch ${colour}`;
}

function renderMap(id, queryPlace, hits) {
  const map = document.querySelector(`#${id}`);
  map.replaceChildren();
  state.data.places.forEach((place) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `place${place.name === queryPlace ? " query" : ""}`;
    button.style.left = `${place.x * 100}%`;
    button.style.top = `${place.y * 100}%`;
    button.title = place.name;
    const label = document.createElement("em");
    label.textContent = shortName(place);
    button.append(label);
    button.addEventListener("click", () => {
      const lost = document.querySelector("#lost-location");
      const found = document.querySelector("#found-location");
      if (id.startsWith("lost") && lost) lost.value = place.name;
      if (id.startsWith("found") && found) found.value = place.name;
      renderMap(id, place.name, hits);
    });
    map.append(button);
  });
  hits.forEach((hit, index) => {
    const pin = document.createElement("span");
    pin.className = "pin";
    pin.textContent = String(index + 1);
    pin.style.left = `calc(${hit.x * 100}% + ${index * 10}px)`;
    pin.style.top = `calc(${hit.y * 100}% + ${index * 6}px)`;
    pin.title = hit.location;
    map.append(pin);
  });
}

function matchCard(hit, mode) {
  const article = document.createElement("article");
  article.className = "match";
  article.append(photoEl(hit), matchBody(hit, mode));
  return article;
}

function photoEl(record) {
  if (record.has_photo) {
    const image = document.createElement("img");
    image.src = `/media/${record.id}.jpg`;
    image.alt = `${record.colour} ${record.category_label}`;
    return image;
  }
  const placeholder = document.createElement("div");
  placeholder.className = "photo-fallback";
  placeholder.textContent = "No photo";
  return placeholder;
}

function matchBody(hit, mode) {
  const body = document.createElement("div");
  const header = document.createElement("header");
  const rank = document.createElement("span");
  rank.className = "rank";
  rank.textContent = `#${hit.rank}`;
  const score = document.createElement("span");
  score.className = "score";
  score.textContent = pct(hit.score);
  header.append(rank, score);
  const title = document.createElement("p");
  title.textContent = `${hit.colour} ${hit.category_label}`;
  const ident = document.createElement("p");
  ident.className = "meta";
  ident.textContent = hit.id;
  const meta = document.createElement("p");
  meta.className = "meta";
  const stamp = hit.found_time ? `found ${when(hit.found_time)}` : `lost ${when(hit.lost_time)}`;
  meta.textContent = `${hit.location} · ${stamp}`;
  body.append(header, title, ident, meta);
  if (hit.note) {
    const note = document.createElement("p");
    note.className = "meta";
    note.textContent = hit.note;
    body.append(note);
  }
  if (mode === "clip") {
    const bars = document.createElement("div");
    bars.className = "bars";
    bars.append(part(hit.visual_part, "#1d6a5f"), part(hit.location_part, "#c4622d"), part(hit.time_part, "#8d3d32"));
    const legend = document.createElement("div");
    legend.className = "legend";
    legend.append(
      chip("#1d6a5f", `visual ${pct(hit.visual_part)}`),
      chip("#c4622d", `place ${pct(hit.location_part)}`),
      chip("#8d3d32", `time ${pct(hit.time_part)}`)
    );
    body.append(bars, legend);
  } else {
    const legend = document.createElement("div");
    legend.className = "legend";
    legend.textContent =
      `Category ${hit.category_match ? "matches" : "differs"}, colour ${hit.colour_match ? "matches" : "differs"}.`;
    body.append(legend);
  }
  return body;
}

function part(value, color) {
  const item = document.createElement("i");
  item.style.width = `${Math.max(0, value) * 100}%`;
  item.style.background = color;
  return item;
}

function chip(color, text) {
  const span = document.createElement("span");
  const dot = document.createElement("i");
  dot.className = "dot";
  dot.style.background = color;
  span.append(dot, document.createTextNode(text));
  return span;
}

function renderResults(prefix, payload) {
  const clip = document.querySelector(`#${prefix}-clip`);
  const linear = document.querySelector(`#${prefix}-linear`);
  clip.className = payload.clip.length ? "" : "empty";
  linear.className = payload.linear.length ? "" : "empty";
  if (!payload.clip.length) {
    clip.textContent = payload.message || "No matching records yet.";
    linear.textContent = payload.message || "No matching records yet.";
    renderMap(`${prefix}-map`, document.querySelector(`#${prefix}-location`).value, []);
    return;
  }
  clip.replaceChildren(...payload.clip.map((hit) => matchCard(hit, "clip")));
  linear.replaceChildren(...payload.linear.map((hit) => matchCard(hit, "linear")));
  if (payload.visual_source === "image") {
    const note = document.createElement("p");
    note.className = "meta";
    note.textContent = "Visual score uses the uploaded photo.";
    clip.prepend(note);
  }
  renderMap(`${prefix}-map`, document.querySelector(`#${prefix}-location`).value, payload.clip);
}

function formPayload(kind) {
  return {
    kind,
    category: document.querySelector(`#${kind}-category`).value,
    colour: document.querySelector(`#${kind}-colour`).value,
    location: document.querySelector(`#${kind}-location`).value,
    event_time: document.querySelector(`#${kind}-time`).value,
    note: document.querySelector(`#${kind}-note`).value,
    image_base64: kind === "lost" ? state.lostPhoto : state.foundPhoto,
  };
}

async function onSubmit(kind, event) {
  event.preventDefault();
  const button = document.querySelector(`#${kind}-btn`);
  const error = document.querySelector(`#${kind}-error`);
  const notice = document.querySelector(`#${kind}-notice`);
  error.hidden = true;
  notice.hidden = true;
  button.disabled = true;
  const hadPhoto = Boolean(kind === "lost" ? state.lostPhoto : state.foundPhoto);
  button.textContent = hadPhoto ? "Encoding photo..." : "Saving...";
  try {
    const response = await fetch("/api/records", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formPayload(kind)),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not save that record.");
    state.data.records = payload.records;
    fillHeader(payload.counts);
    renderRecords();
    notice.hidden = false;
    notice.textContent = `Saved ${payload.record.id}. Open Database to see every live record.`;
    if (payload.matches) renderResults(kind, payload.matches);
    clearPhoto(kind);
  } catch (err) {
    error.hidden = false;
    error.textContent = err.message;
  } finally {
    button.disabled = false;
    button.textContent = kind === "lost" ? "Save and find matches" : "Save found item";
  }
}

function clearPhoto(kind) {
  if (kind === "lost") state.lostPhoto = "";
  else state.foundPhoto = "";
  const input = document.querySelector(`#${kind}-photo`);
  const name = document.querySelector(`#${kind}-photo-name`);
  if (input) input.value = "";
  if (name) name.textContent = "No photo selected";
}

function renderRecords() {
  const items = state.data.records.filter((record) => {
    const kindOk = state.filterKind === "all" || record.kind === state.filterKind;
    const categoryOk = state.filterCategory === "all" || record.category === state.filterCategory;
    const colourOk = state.filterColour === "all" || record.colour === state.filterColour;
    return kindOk && categoryOk && colourOk;
  });
  const counts = state.data.records.reduce(
    (acc, record) => {
      acc[record.kind] += 1;
      return acc;
    },
    { lost: 0, found: 0 }
  );
  document.querySelector("#records-count").textContent =
    items.length === 0
      ? "No records yet. Report a lost or found item to start the live database."
      : `${items.length} shown · ${counts.lost} lost · ${counts.found} found`;
  const grid = document.querySelector("#records");
  if (!items.length) {
    grid.replaceChildren();
    const empty = document.createElement("p");
    empty.className = "empty wide-empty";
    empty.textContent = "The training photos stay with the model. This page only lists items people upload here.";
    grid.append(empty);
    return;
  }
  grid.replaceChildren(
    ...items
      .slice()
      .reverse()
      .map((record) => {
        const card = document.createElement("article");
        card.className = "item";
        card.append(photoEl(record));
        const body = document.createElement("div");
        const badge = document.createElement("span");
        badge.className = `badge ${record.kind}`;
        badge.textContent = record.kind;
        const ident = document.createElement("span");
        ident.className = "ident";
        ident.textContent = record.id;
        const title = document.createElement("strong");
        title.textContent = `${record.colour} ${record.category_label}`;
        const meta = document.createElement("span");
        const stamp = record.found_time ? when(record.found_time) : when(record.lost_time);
        meta.textContent = `${record.location} · ${stamp}`;
        body.append(badge, ident, title, meta);
        if (record.note) {
          const note = document.createElement("span");
          note.textContent = record.note;
          body.append(note);
        }
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "danger";
        remove.textContent = "Delete";
        remove.addEventListener("click", () => onDelete(record.id));
        body.append(remove);
        card.append(body);
        return card;
      })
  );
}

async function onDelete(itemId) {
  const password = document.querySelector("#admin-password").value;
  if (!password) {
    alert("Enter the admin password first.");
    document.querySelector("#admin-password").focus();
    return;
  }
  if (!window.confirm(`Delete ${itemId}? This cannot be undone.`)) return;
  const response = await fetch("/api/records/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: itemId, password }),
  });
  const payload = await response.json();
  if (!response.ok) {
    alert(payload.error || "Could not delete that record.");
    return;
  }
  state.data.records = payload.records;
  fillHeader(payload.counts);
  renderRecords();
}

document.querySelectorAll("nav button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("nav button").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
    button.classList.add("active");
    document.querySelector(`#view-${button.dataset.view}`).classList.add("active");
  });
});

document.querySelector("#lost-form").addEventListener("submit", (event) => onSubmit("lost", event));
document.querySelector("#found-form").addEventListener("submit", (event) => onSubmit("found", event));
document.querySelector("#lost-clear-photo").addEventListener("click", () => clearPhoto("lost"));
document.querySelector("#found-clear-photo").addEventListener("click", () => clearPhoto("found"));

function shrink(file) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const url = URL.createObjectURL(file);
    image.onload = () => {
      const scale = Math.min(1, 512 / Math.max(image.width, image.height));
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(image.width * scale));
      canvas.height = Math.max(1, Math.round(image.height * scale));
      canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      resolve(canvas.toDataURL("image/jpeg", 0.86));
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not read that photo."));
    };
    image.src = url;
  });
}

boot().catch((error) => {
  document.querySelector("main").textContent = error.message;
});
