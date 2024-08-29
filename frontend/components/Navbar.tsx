import Link from "next/link";

export default function Navbar() {
  return (
    <nav style={{ padding: "1rem", backgroundColor: "#333", color: "white" }}>
      <h1>My Website</h1>
      <ul style={{ display: "flex", listStyle: "none", gap: "1rem" }}>
        <li>
          <Link href="/" style={{ color: "white", textDecoration: "none" }}>
            Home
          </Link>
        </li>
        <li>
          <Link href="/chat" style={{ color: "white", textDecoration: "none" }}>
            Chat
          </Link>
        </li>
        <li>
          <Link
            href="/contact"
            style={{ color: "white", textDecoration: "none" }}
          >
            Stored PDFs
          </Link>
        </li>
      </ul>
    </nav>
  );
}
