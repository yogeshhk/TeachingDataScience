const BIRTHDAY = new Date(1973, 2, 27, 11, 25, 0); // March 27 1973, 11:25 AM
const LIFESPAN_YEARS = 100;
const END_DATE = new Date(BIRTHDAY);
END_DATE.setFullYear(END_DATE.getFullYear() + LIFESPAN_YEARS);

const MS_PER_DAY = 1000 * 60 * 60 * 24;

function update() {
  const now = new Date();

  const msDone    = now - BIRTHDAY;
  const msLeft    = END_DATE - now;

  const daysDone  = Math.floor(msDone  / MS_PER_DAY);
  const daysLeft  = Math.max(0, Math.floor(msLeft / MS_PER_DAY));
  const totalDays = Math.floor((END_DATE - BIRTHDAY) / MS_PER_DAY);

  const pct = Math.min(100, (daysDone / totalDays) * 100).toFixed(2);

  const years  = Math.floor(daysDone / 365.25);
  const months = Math.floor((daysDone % 365.25) / 30.44);
  const days   = Math.floor(daysDone % 30.44);

  document.getElementById("done").textContent  = daysDone.toLocaleString();
  document.getElementById("left").textContent  = daysLeft.toLocaleString();
  document.getElementById("total").textContent = totalDays.toLocaleString();
  document.getElementById("pct").textContent   = pct + "%";
  document.getElementById("age").textContent   = `${years}y ${months}m ${days}d`;
  document.getElementById("bar").style.width   = pct + "%";
  document.getElementById("bar").textContent   = pct + "%";

  const endStr = END_DATE.toLocaleDateString("en-IN", { day: "numeric", month: "long", year: "numeric" });
  document.getElementById("enddate").textContent = endStr;
}

update();
setInterval(update, 60000); // refresh every minute
